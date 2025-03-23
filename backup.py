#!/usr/bin/python3

options = {
    # metadata_dir is where imported configs are stored. It can be an
    # absolute path, or a path relative to this program's location
    "metadata_dir": "metadata",
    # verbose is here so it can have a value at program-initialization time;
    # gets overwritten by cli opts early on
    "verbose": True,
    "always-prime": True,
}

VERSION = 0
PROGRAM_DESCRIPTION = "a backups and snapshots tool for QubesOS+ZFS systems"
ERROR_CODES = {
    "input": 1,
    "fatal": 2,# all fully-unhandled errors go here
    "yaml": 3,
    "notfound": 4,# only when specifying non-existant files on the command line
    "receiver": 7,
    "send": 8,
}
CONFIG_FILE_VERSION = "1"
CONFIG_FILE_EXT = "backupcfg"
PRIMER_SUFFIX = "-primer"

# libraries reference:
# qubesadmin:
# https://dev.qubes-os.org/projects/core-admin-client/en/latest/py-modindex.html
# libzfs_core:
# https://pyzfs.readthedocs.io/en/latest/index.html
#
import sys, traceback, itertools, os, os.path, time, \
    argparse, re, subprocess, datetime, json
import types, collections, collections.abc, functools
from collections.abc import Callable
import asyncio
chain_iter = itertools.chain.from_iterable
#
import qubesadmin
qapp = qubesadmin.Qubes()
import libzfs_core

colours = types.SimpleNamespace(
    OK = '\x1b[32m', #green
    ERR = '\x1b[31m', #red
    WARN = '\x1b[93m', #bright yellow
    BRIGHT_YELLOW = '\x1b[93m',
    YELLOW = '\x1b[33m',
    GREEN = '\x1b[32m',
    RED = '\x1b[31m',
    BLUE = '\x1b[34m',
    BRIGHT_BLUE = '\x1b[94m',
    BRIGHT_BLACK = '\x1b[90m',
    BOLD = '\x1b[1m',
    RESET = '\x1b[0m'
)

apparent_command_name = os.path.basename(sys.argv[0])
real_command_dir, real_command_name = \
    os.path.split(os.path.realpath(sys.argv[0], strict=True))
if not options['metadata_dir'].startswith('/'):
    options['metadata_dir'] = os.path.join(real_command_dir, options['metadata_dir'])



warning_counter = 0
def warn(msg, prefix="Warning: ", count=1):
    global warning_counter
    warning_counter += count
    print(f"{colours.WARN}{prefix}{msg}{colours.RESET}", file=sys.stderr)

def print_bg_msg(msg):
    if options['verbose']:
        print(f"{colours.BRIGHT_BLACK}{msg}{colours.RESET}", file=sys.stderr)

def track_commandline(args):
    cmdline = ' '.join([(f'"{a}"' if ' ' in a else a) for a in args])
    print_bg_msg(cmdline)

def run_cmd(*args, **kwargs):
    opts = {"capture_output":True, "check":False, "text":True} | kwargs
    track_commandline(args)
    p = subprocess.run(args, **opts)

    if p.stderr:
        print_bg_msg(p.stderr)
    try:
        p.check_returncode()
    except:
        if p.stdout:
            print_bg_msg(p.stdout)
        raise

    return p.stdout

def human_readable_bytesize(num:int, field_w=18):
    #I just needed something that works ok?
    def commas(intd):
        parts = []
        while len(intd) > 3:
            intd,p = intd[:-3],intd[-3:]
            parts.append(p)
        parts.append(intd)
        return ','.join(reversed(parts))
    if num < 0:
        raise ValueError()
    elif num < 10_000_000:
        eh = commas(str(num))
        return eh.rjust(field_w)
    elif num < 1_100_000_000:
        a = num / (1024**2)
        b = round(a)
        # b = round(a , (1 if a < 10 else 0))
        c = commas(str(b)) + " M-- ---"
        return c.rjust(field_w)
    else:
        a = num / (1024**3)
        b = round(a, (1 if a < 10 else 0))
        if b >= 10:
            b = round(b)
        c = str(b)
        intd,dec,fracd = c.partition('.')
        intd = commas(intd)
        if dec == '.':
            x = f"{intd}.{fracd[:1]} G --- ---"
        else:
            x = intd + " G-- --- ---"
        return x.rjust(field_w)

class AdvanceableChainIter():
    def __init__(self, *its):
        self.stop = None
        self.its = iter(its)
        self.advance()

    def advance(self):
        if self.stop is not None:
            return
        try:
            self.cur_item = next(self.its)
        except StopIteration as e:
            self.stop = e
        else:
            self.cur_it = iter(self.cur_item)

    def __iter__(self):
        return self

    def __next__(self):
        while self.stop is None:
            try:
                return next(self.cur_it)
            except StopIteration:
                self.advance()
        raise self.stop

class MyError(Exception):
    def __init__(self, *args, error_code=None, **kwargs):
        super().__init__(*args, **kwargs)
        if type(error_code) is str:
            self.error_code = ERROR_CODES[error_code]
        elif error_code is not None:
            self.error_code = error_code

    def filtered_args(self):
        fmt = {
            True: traceback.format_exception,
            False: traceback.format_exception_only
        }[options['verbose']]
        for a in self.args:
            if isinstance(a, Exception):
                for s in fmt(a):
                    yield s.rstrip('\n')
            else:
                yield a

    def __str__(self):
        if not hasattr(self, "args_str"):
            self.args_str = '\n'.join(str(a) for a in self.filtered_args())
        return self.args_str

class ConfigError(MyError):
    error_code = ERROR_CODES['input']

    def __init__(self):
        self.verrs = []

    def __iadd__(self, other):
        assert hasattr(other,"__len__") and len(other) == 2
        self.verrs.append(other)
        return self

    def __str__(self):
        summary = [f"Config is invalid. There were {len(self.verrs)} validation error(s):"]
        formatted_verrs = (f"! [{k}]: {e}" for k,e in self.verrs)
        return '\n'.join(itertools.chain(summary,formatted_verrs))

    def __bool__(self):
        return bool(self.verrs)

class ReceiverError(MyError):
    error_code = ERROR_CODES['receiver']

class ConfigHeader(collections.abc.Mapping):
    FIELD_TERM_CHAR = b';'
    CHAIN_TERM_CHAR = b'\n'
    VALUE_CHARS = set(range(0x20,0x80)) - set(FIELD_TERM_CHAR+CHAIN_TERM_CHAR)
    FILE_SIG = [
        (2, "bu"),
        (4, "ver"),
    ]
    FIELDS = [
        (2, "enc"),
        (16, "importdate"),
        (2, "state"),
        (21, "comment"),
    ]
    FILE_SIG_CUR_VERSION = {
        "bu": "bu",
        "ver": CONFIG_FILE_VERSION,
    }

    def __init__(self):
        self.sig = self.FILE_SIG_CUR_VERSION.copy()
        self.info = dict((name,"") for sz,name in self.FIELDS)

    def __contains__(self, k):
        return k in self.info

    def __iter__(self):
        return iter(self.info)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, k):
        return self.info[k]

    def __setitem__(self, k, v):
        self.info[k] = self._validate_value(self.FIELDS, k, v)

    def trunc(self, k, v):
        self.info[k] = self._validate_value(self.FIELDS, k, v, truncate=True)

    @classmethod
    def read_chain(cls, f):
        o = cls()
        o.sig = cls._read_header(cls.FILE_SIG, f)
        if o.sig != cls.FILE_SIG_CUR_VERSION:
            raise Exception("unrecognized file header")
        o.info = cls._read_header(cls.FIELDS, f)
        if f.read(1) != cls.CHAIN_TERM_CHAR:
            raise Exception("corrupt header")
        return o

    def write_chain(self, f):
        self._write_header(self.FILE_SIG, f, self.FILE_SIG_CUR_VERSION)
        self._write_header(self.FIELDS, f, self.info)
        f.write(b'\n')

    #

    def _write_header(self, defin, f, obj):
        for sz,name in defin:
            save_v = obj[name].encode('ascii', errors='strict')
            f.write(save_v.ljust(sz, self.FIELD_TERM_CHAR))

    @staticmethod
    def _getfielddef(defin, k):
        for i,(sz,name) in enumerate(defin):
            if name == k:
                return (i,sz)
        raise KeyError()

    @classmethod
    def _validate_value(cls, defin, k, v, truncate=False):
        _,field_sz = cls._getfielddef(defin, k)
        saved_v = str.encode(v, 'ascii', errors='strict')
        if not (set(saved_v) <= cls.VALUE_CHARS):
            raise ValueError("invalid characters")
        if truncate:
            v = saved_v[:field_sz].decode('ascii')
        elif len(saved_v) > field_sz:
            raise ValueError("string too long")
        return v

    @classmethod
    def _read_header(cls, defin, f):
        h_sz = sum(next(zip(*defin)))
        data = f.read(h_sz)
        if len(data) != h_sz:
            raise Exception("header could not be read")
        items = []
        offset = 0
        for sz,k in defin:
            v = data[offset:offset+sz].split(cls.FIELD_TERM_CHAR)[0].decode("ascii")
            items.append((k,v))
            offset += sz
        return dict(items)

class Config():
    PERSISTENT = ('user_config', 'raw_user_config', 'objects')

    @classmethod
    def import_(cls, filename):
        try:
            with open(filename) as f:
                raw_user_config = f.read()
        except FileNotFoundError as e:
            err_str = f"{e.strerror}: {e.filename}"
            raise MyError(err_str, error_code='notfound')

        user_config = cls._parse_user_config(raw_user_config)
        cls._check_objs_exist(user_config)
        objects = [cls._mk_objects_pt2(o) for o in cls._mk_objects_pt1_gen(user_config)]

        o = cls()
        vars(o).update((k,v) for k,v in locals().copy().items() if k in cls.PERSISTENT)
        o.header = ConfigHeader()
        o.header["importdate"] = \
            datetime.datetime.now().isoformat(sep=' ',timespec='minutes')

        o._serialize()
        o.get_graph()
        for ds_info in o.included_datasets:
            o._check_on_snapshot_state(ds_info)
        print(f"Config validation {colours.OK}PASSED{colours.RESET}")
        time.sleep(0.9)
        return o

    def save(self):
        with open(self.filepath(self.name), 'wb') as f:
            self.header.write_chain(f)
            f.write(self._serialize())

    @classmethod
    def load(cls, name):
        o = cls()
        with open(cls.filepath(name), 'rb') as f:
            o.header = ConfigHeader.read_chain(f)
            serialized_data = f.read()
        data = json.loads(serialized_data.decode('ascii', errors='strict'))
        vars(o).update(data)
        return o

    @property
    def name(self):
        return self.user_config["name"]

    @staticmethod
    def filepath(name):
        return os.path.join(options["metadata_dir"], name + '.' + CONFIG_FILE_EXT)

    @classmethod
    def all_config_names(cls):
        if not os.path.isdir(options["metadata_dir"]):
            return set()
        files = os.listdir(options["metadata_dir"])
        return set(fn.rsplit('.',1)[0] for fn in files if fn.endswith('.'+CONFIG_FILE_EXT))

    @classmethod
    def list_configs(cls):
        r = []
        for name in cls.all_config_names():
            with open(cls.filepath(name), 'rb') as f:
                header = ConfigHeader.read_chain(f)
            row = (header["importdate"], name, header["state"], header["comment"])
            r.append(row)
        r.sort()
        return r

    @functools.cached_property
    def included_datasets(self):
        x = (o['datasets'] for o in self.objects if o['role'] == 'include')
        return list(chain_iter(x))

    def get_graph(self):
        try:
            r = self._graph
        except:
            r = self._graph = self._make_graph(self.objects)
        return r

    def print_import_analysis(self):
        print("Analysis:")
        self._print_coverage_analysis(self.get_graph())
        print()
        self._print_snapshots_analysis()

    def do_snapshots(self):
        def ds_gen():
            for o in self.objects:
                if o['role'] == 'include':
                    for ds_info in o["datasets"]:
                        yield (ds_info['ds'], ds_info, o)
        #
        deferred = 0
        target = self.user_config['scope']['target-snapshot']
        snaps_allowed = self.user_config['scope']['allow-snapshot-creation']
        #
        for ds,ds_info,o in ds_gen():
            if ds_info['snap'] in ('unknown','blocked'):
                self._check_on_snapshot_state(ds_info)

            # can we snap?
            if not snaps_allowed:
                continue
            if ds_info['snap'] != 'would':
                continue
            if o['type'] == 'qubes':
                args = f"qvm-check -q --running {o['member_name']}".split()
                exitcode = subprocess.run(args).returncode
                if exitcode == 0:
                    warn(f"Qube {o['member_name']} is running. Deferring snapshot creation.")
                    deferred += 1
                    continue
            elif o['type'] == 'datasets':
                zvols = contained_zvols_attached_to_any_domain(ds)
                if zvols:
                    for zvol in zvols:
                        warn(f"Volume {zvol} is attached to a qube. Deferring snapshot creation.")
                    deferred += 1
                    continue

            # yes, snap
            snap_name = ds+target
            primer_name = snap_name+PRIMER_SUFFIX
            if self.uses_primer or options['always-prime']:
                if not libzfs_core.lzc_exists(primer_name.encode('ascii')):
                    run_cmd("zfs","snapshot",primer_name)
                ds_info['primer_label'] = target.removeprefix('@') + PRIMER_SUFFIX
            run_cmd("zfs","snapshot","-r",snap_name)
            ds_info['snap'] = 'done'
            print("snapshotted "+ds)

        self._print_snapshots_analysis()
        if deferred:
            warn(f"{deferred} deferred snapshots. Re-run the command once " \
                "the preconditions are satisfied.", prefix='', count=0)

    @property
    def uses_primer(self):
        return self.user_config['scope']['since'] == 'beginning' \
            and self.user_config['scope']['progressive'] == False

    def calculate_send(self):
        calc_count = 0
        unsnapped_count = 0
        send_cmd_makers = self._make_send_cmd_makers('calc')

        for ds_info in self.included_datasets:
            if type(ds_info['size']) is int:
                continue
            if ds_info['snap'] not in ('done','yes'):
                unsnapped_count += 1
                continue

            @functools.cache
            def receiver():
                return Receiver(**self.user_config['receiver'], user="user")

            sz = 0
            for mk_cmd in send_cmd_makers:
                if mk_cmd.is_primer:
                    # will we actually need to send it?
                    dest = f"{receiver().dest_dataset}/{ds_info['ds']}@{ds_info['primer_label']}"
                    if receiver().dataset_exists(dest):
                        continue
                out = run_cmd(*mk_cmd(**ds_info))
                for l in out.splitlines():
                    fields = l.split()
                    if fields[:1] == ["size"]:
                        sz += int(fields[1])
                        break
                else:
                    raise Exception("size not found")
            ds_info['size'] = sz
            ds_info['send_status'] = 'ready'
            calc_count += 1

        if calc_count:
            print(f"Calculated size of {calc_count} objects")
        if unsnapped_count:
            warn(f"{unsnapped_count} snapshots are missing. Calculations for " \
                "those objects have been skipped.")
        self._print_calc_result()

    def do_send(self, *, complete_send_only=True):
        total_bytes = 0
        initial_sent_bytes = 0
        datasets_ready = []
        datasets_interrupted = []
        for ds_info in self.included_datasets:
            s = ds_info['send_status']
            if ds_info['send_status'] != 'notready':
                total_bytes += ds_info['size']
                match ds_info['send_status']:
                    case 'ready':
                        datasets_ready.append(ds_info)
                    case 'interrupted':
                        datasets_interrupted.append(ds_info)
                    case 'done':
                        initial_sent_bytes += ds_info['size']
            elif complete_send_only:
                raise Exception("can't send: previous phases are incomplete")

        datasets_iter = AdvanceableChainIter(datasets_interrupted, datasets_ready)
        local_send_cmd_makers = self._make_send_cmd_makers('send')
        exceptions = []
        receiver = Receiver(**self.user_config['receiver'])
        transfer_metrics = receiver.new_transfer_metrics()
        transfer_metrics['bytes_sent'] = initial_sent_bytes

        def progress_backspace(flush=False):
            print('\r', ' '*40, '\r', sep='', end='', flush=flush)

        def progress_print(*args):
            progress_backspace()
            print(transfer_metrics['bytes_sent'], '/', total_bytes,
                end='', flush=True)

        # desired free space kinda arbitrary; I just picked what felt right
        # idk what kind of storage overhead actually exists, if any
        if (int(total_bytes * 1.02) + 2**17) > receiver.get_space_avail():
            raise ReceiverError("Receiver does not have enough free space")

        for ds_info in datasets_iter:
            for mk_local_cmd in local_send_cmd_makers:
                local_cmd = mk_local_cmd(**ds_info)

                if mk_local_cmd.is_primer:
                    snap_segment = '@' + ds_info['primer_label']
                else:
                    snap_segment = self.user_config['scope']['target-snapshot']
                dest_snap = receiver.dest_dataset +'/'+ ds_info['ds'] + snap_segment
                dest_parent, dest_snap_shortname = dest_snap.rsplit('/', maxsplit=1)

                try:
                    if receiver.dataset_exists(dest_snap):
                        if mk_local_cmd.is_primer:
                            continue
                        else:
                            raise ReceiverError("Destination snapshot already exists")

                    # create parent container if it doesn't exist
                    receiver.run_cmd(*("zfs create -p -u".split()), dest_parent)

                    # send! HOORAY!
                    print("sending", local_cmd[-1])
                    send = receiver.stream_to(
                        local_cmd,
                        ["zfs", "receive", "-e", "-u", dest_parent],
                        transfer_metrics, progress_print
                    )
                    asyncio.run(send)

                    if options['verbose']:
                        print("\nOK")
                    else:
                        time.sleep(0.2)
                        progress_backspace(True)

                except Exception as e:
                    print(f"{colours.ERR}{e}{colours.RESET}", file=sys.stderr)
                    exceptions += [f"error while attempting to send {ds_info['ds']}", e]
                    ds_info['send_status'] = 'interrupted'
                    #
                    datasets_iter.advance()
                    break

            ds_info['send_status'] = 'done'

        sent = transfer_metrics['bytes_sent'] - initial_sent_bytes
        if sent:
            print("sent " + human_readable_bytesize(sent).strip(' -'))
        if exceptions:
            print(f"{colours.ERR}send failed{colours.RESET}", file=sys.stderr)
            raise MyError(*exceptions, error_code='send')
        else:
            print("send completed successfully!")

    @staticmethod
    def _parse_user_config(raw_config):
        try:
            import yaml
            config = yaml.safe_load(raw_config)
        except Exception as e:
            raise MyError("yaml error", e, error_code='yaml')

        validation_errors = ConfigError()
        class VErr(Exception): pass
        def _inner_test(keys, strongtype=None, cat=None, validator=None):
            x = config
            for k in keys.split('.'):
                try:
                    x = x[k]
                except KeyError:
                    if cat == "group":
                        raise VErr("Group is not defined")
                    else:
                        raise VErr("Missing required key")
            if strongtype not in (None, type(x)):
                raise VErr(f"Value must be type {strongtype.__name__}, but {type(x).__name__} was found.")
            match cat:
                case 'snap':
                    if not re.fullmatch(r'(@[-\w]+|beginning)', x):
                        raise VErr("Doesn't look like a snapshot name")
                case 'word':
                    if not re.fullmatch(r'[-\w]+', x):
                        raise VErr("Contains invalid characters")
                case 'dataset':
                    if not re.fullmatch(r'\w[-\w]*(/?[-\w]+)*', x):
                        raise VErr("Doesn't look like a dataset")
            if validator is not None and not validator(x):
                raise VErr("Failed validation")

        def test(keys, *args, **kwargs):
            nonlocal validation_errors
            try:
                _inner_test(keys, *args, **kwargs)
            except VErr as e:
                validation_errors += (keys, e.args[0])
                return False
            return True

        # To truly validate things, we need to cross reference everything
        # to real system objects (qubes, snapshots, etc.), which doesn't
        # happen til later, so we can't be perfect here.
        # The validation here is (mostly) to benefit the user with better
        # error messages.
        test("scope.target-snapshot", str, "snap", lambda x: x.startswith("@"))
        test("name", str, cat="word")
        test("scope.since", str, cat="snap")
        test("scope.progressive", bool)
        test("scope.allow-snapshot-creation", bool)
        test("receiver.qube", str, cat="word")
        test("receiver.dataset", str, cat="dataset")
        if test("include-groups", list) and test("exclude-groups", list):
            group_types = ("qubes","datasets")
            groups = list(itertools.chain(config["include-groups"],config["exclude-groups"]))
            groups_set = set(groups)
            for g in groups_set:
                if test(g, cat="group"):
                    test(g+".type", validator=lambda x: x in group_types)
                    test(g+".members", list)
            if len(groups) != len(groups_set):
                validation_errors += "groups", "Includes/excludes contain redundant group references"

        if validation_errors:
            raise validation_errors
        #else
        return config

    @staticmethod
    def _check_objs_exist(user_config):
        """check if all referenced qubes and datasets exist
        (except for receiver.dataset and all things in unused groups)"""
        def all_of(t):
            for g in itertools.chain(user_config["include-groups"],user_config["exclude-groups"]):
                if user_config[g]["type"] == t:
                    yield from user_config[g]["members"]
        errs = ConfigError()
        if user_config['receiver']['qube'] not in qube_list.get():
            errs += ('receiver.qube', "Qube does not exist")
        for q in all_of("qubes"):
            if q not in qube_list.get():
                errs += ("qube:"+q, "Qube does not exist")
        for ds in all_of("datasets"):
            if ds not in dataset_list.get():
                errs += ("dataset:"+q, "Dataset does not exist")
        if errs:
            raise errs

    @staticmethod
    def _mk_objects_pt1_gen(user_config):
        """atomize include/exclude targets into standalone objects,
        and resolve all datasets"""
        for r in ("include","exclude"):
            for gn in user_config[f"{r}-groups"]:
                for m in user_config[gn]["members"]:
                    yield {
                        "role": r,
                        "groupname": gn,
                        "member_name": m,
                        "type": user_config[gn]["type"],
                        "ignore_datasets": [],
                    }

    @staticmethod
    def _mk_objects_pt2(o):
        m = o["member_name"]
        if o['type'] == "qubes":
            o["name"] = "qube:" + m
            vols = qubesd_query("admin.vm.volume.List",m)
            vol_parents = set()
            vol_pools = set()
            infos = []
            for vol in vols:
                if vol == "kernel":
                    continue
                vol_info = qubesd_query("admin.vm.volume.Info",m,vol)
                if vol_info["ephemeral"] != "False":
                    raise Exception(["idk what ephemeral does",m,vol,vol_info])
                if vol_info["pool"] not in qubes_pools_info.get():
                    warn(f"not all of qube {m}'s volumes are in "
                        "an applicable pool. Some volumes won't be backed up")
                    continue
                if vol_info["save_on_stop"] != "True":
                    o['ignore_datasets'].append(vol_info["vid"])
                    continue
                vol_parent, vol_name = vol_info["vid"].rsplit('/',maxsplit=1)
                assert vol == vol_name
                vol_parents.add(vol_parent)
                vol_pools.add(vol_info["pool"])
                infos.append(vol_info)
            if 1 == len(vol_parents) == len(vol_pools):
                #all volumes share the same parent, use the parent for backup
                dataset_names = [next(iter(vol_parents))]
            else:
                dataset_names = [i["vid"] for i in infos]
        elif o['type'] == "datasets":
            o["name"] = "dataset:" + m
            dataset_names = [m]
        else:
            assert False
        if o['role'] == "include":
            dataset_info = {
                # "ds": <dataset name>,
                "snap": "unknown",
                "size": None,
                "send_status": "notready",
                "primer_label": None,
            }
        else:
            dataset_info = {}
        o['datasets'] = [(dataset_info | {"ds":n}) for n in dataset_names]
        return o

    @staticmethod
    def _make_graph(objects):
        """make graph, checking for duplicate and/or nested qubes and datasets"""
        @(lambda x: dict(x()))
        def graph():
            qubes_containers = '|'.join(pi["container"] for pi in qubes_pools_info.get().values())
            disp_pat = re.compile('(' + qubes_containers + r')/disp\d{1,4}')
            for ds in dataset_list.get():
                ignore = bool(disp_pat.fullmatch(ds)) or '/.' in ds
                yield ds,{"own_obj": None, "descendant_objects": [], "ignore": ignore}
        #
        errs = ConfigError()
        for o in objects:
            for i,ds_info in enumerate(o['datasets']):
                ds = ds_info['ds']
                node = graph[ds]
                if node["own_obj"] is not None:
                    msg = f"Duplicate reference conflicts with {node['own_obj'][0]['name']}"
                    errs += o['name'], msg
                # parents = [(pds,graph[pds]) for pds in parents_iter(ds)]
                ancestor_objs = filter(bool, (graph[pds]["own_obj"] for pds in zfs_ancestors_iter(ds)))
                ancestor_collisions = ((a,(o,i)) for a in ancestor_objs)
                descendant_collisions = (((o,i),d) for d in node["descendant_objects"])
                for a,d in itertools.chain(ancestor_collisions, descendant_collisions):
                    msg = f"{d[0]['name']} is nested in {a[0]['datasets'][a[1]]['ds']}"
                    errs += a[0]['name'], msg

                node["own_obj"] = (o,i)
                for pds in zfs_ancestors_iter(ds):
                    graph[pds]["descendant_objects"].append((o,i))

            for ids in o["ignore_datasets"]:
                if ids in graph:
                    graph[ids]["ignore"] = True
        if errs:
            raise errs
        return graph

    @staticmethod
    def _print_coverage_analysis(graph):
        # analyze coverage (warn for qubes/datasets neither included nor excluded)
        excluded = []
        included = []
        unspecified = []
        ignores = 0
        irrelevants = 0
        for ds,info in sorted(graph.items()):
            if info["own_obj"] is not None:
                match info["own_obj"][0]["role"]:
                    case "include":
                        included.append(ds)
                    case "exclude":
                        excluded.append(ds)
            else:
                if not info["descendant_objects"]:
                    try:
                        pds = next(zfs_ancestors_iter(ds))
                    except StopIteration:
                        pass
                    else:
                        if not graph[pds]["descendant_objects"]:
                            irrelevants += 1
                            continue
                    if info["ignore"]:
                        ignores += 1
                    else:
                        unspecified.append(ds)
                else:
                    irrelevants += 1
        global warning_counter
        warning_counter += len(unspecified)

        print("\nIncluded datasets:")
        posi = f"{colours.GREEN}+{colours.RESET}"
        print(posi,f'\n{posi} '.join(included))

        print("\nExcluded datasets:")
        nega = f"{colours.RED}-{colours.RESET}"
        print(nega,f'\n{nega} '.join(excluded))

        print(f"\n{colours.BRIGHT_YELLOW}Warning: The following datasets have not been " \
            "referenced. Consider explicitly including or excluding them:")
        print('-','\n- '.join(unspecified),f"{colours.RESET}")

        print(f"\nAdditionally:\n{ignores} datasets have been automatically ignored " \
            f"by the program's built-in rules.\n{irrelevants} datasets have " \
            "been implicitly ignored based on the includes/excludes.")

    def _check_on_snapshot_state(self, ds_info, reset=False):
        if 'snap' not in ds_info:
            raise TypeError("There is no snapshot state associated with "\
                "that dataset. The function might have been mistakenly " \
                "called on an excluded object.")
        if ds_info['snap'] in ('yes','done') and not reset:
            return
        ds = ds_info['ds']
        target = self.user_config['scope']['target-snapshot'][1:]
        args = f"zfs list -H -t snapshot -o name -s createtxg -r {ds}".split()
        snapshots = run_cmd(*args).splitlines()
        has_snap = 'no'
        prev_label = None
        for snap in snapshots:
            sds,sep,label = snap.partition('@')
            assert sep == '@'
            if label == target:
                if sds == ds:
                    has_snap = 'yes'
                    break
                else:
                    has_snap = 'descendant'
            prev_label = label
        ds_info['snap'] = {'no':'would', 'descendant':'blocked', 'yes':'yes'}[has_snap]
        if has_snap == 'yes':
            if prev_label is None and self.uses_primer:
                ds_info['snap'] = 'blocked'
                warn(f"There are no previous snapshots to {ds}@{target} (to use " \
                    "as a primer). This snapshot will remain blocked until it is" \
                    "recreated with a primer.")
            # it is not ideal when prev_label is a user-created snap that was
            # not intended to be used as the primer. If it is very different
            # from the target-snap we may end up sending a lot of unwanted
            # data. But the result is acceptable in most cases, so let's not
            # worry about it rn
            ds_info['primer_label'] = prev_label

    def _print_snapshots_analysis(self):
        count = collections.Counter()
        for ds_info in self.included_datasets:
            count[ds_info['snap']] += 1
        if count['unknown'] > 0:
            if count.total() == count['unknown']:
                warn("Requested snapshots analysis results but no analysis has been done")
            else:
                warn("Refusing to print incomplete snapshot analysis results. " \
                    f"({count['unknown']} objects would be omitted)")
            return

        print(f"{count['yes']+count['done']}/{count.total()} required snapshots currently exist.")
        if self.user_config['scope']['allow-snapshot-creation']:
            print(f"{count['would']} snapshots are able to be automatically created.")
        elif count['would'] + count['blocked'] > 0:
            print("scope.allow-snapshot-creation is disabled and so snapshots " \
                "must be created manually.")
        if count['unknown'] > 0:
            warn(f"Incomplete snapshot analysis: {count['unknown']} objects were omitted.")

    def _serialize(self):
        obj = dict((name, getattr(self,name)) for name in self.PERSISTENT)
        return json.dumps(obj, indent=2).encode('ascii', errors='strict')

    def _make_send_cmd_makers(self, mode):
        since, progressive, target = [self.user_config['scope'][k] for k in \
            ('since','progressive','target-snapshot')]
        beginning = since == 'beginning'

        base = ["zfs","send"]
        mode_args = {'calc':["--dryrun","-P"], 'send':[]}[mode]
        # primer args
        a0 = ["-p", "{ds}@{primer_label}"]
        # target-snapshot args
        a1 = ["-R"]
        if not (beginning and progressive):
            a1 += [("-i","-I")[int(progressive)]]
            a1 += ["{ds}" + (since,"@{primer_label}")[int(beginning)]]
        a1 += ["{ds}"+target]
        command_templs = [(base+mode_args+a) for a in (a0,a1)]

        def make(phase, templ):
            def f(**ctx):
                cmd = [arg.format(**ctx) for arg in templ]
                return cmd
            f.phase = phase
            f.is_primer = phase == 0
            return f

        maker_specs = list(enumerate(command_templs))[int(not self.uses_primer):]
        return [make(i,templ) for i,templ in maker_specs]

    def _print_calc_result(self):
        total = 0
        skipped = 0
        print("\nsend size:")
        for ds_info in self.included_datasets:
            sz = ds_info['size']
            if type(sz) is int:
                print(human_readable_bytesize(sz), ds_info['ds'])
                total += sz
            else:
                skipped += 1
        if not skipped:
            print(human_readable_bytesize(total),"(total)")
        else:
            warn("There are uncalculated datasets. Results are incomplete.")

class Receiver():
    OUTPUT_ARGS = ["--pass-io", "--no-colour-output",
        "--no-colour-stderr", "--filter-escape-chars"]
    NULL_DATASET = object()

    def __init__(self, qube, dataset, user="root"):
        self.qube = qube
        self.dest_dataset = dataset
        self.user = user
        try:
            if not qapp.domains[self.qube].is_running():
                raise ReceiverError(self.qube+" isn't running")
        except KeyError:
            raise ReceiverError("receiver qube doesn't exist!")
        if not (dataset is self.NULL_DATASET or self.dataset_exists(dataset)):
            raise ReceiverError("Unable to confirm destination dataset " \
                f"({self.dest_dataset}) exists in receiver qube.")

    def run_cmd(self, *cmd, **kwargs):
        if len(cmd) == 0:
            raise ValueError()
        args = ["qvm-run", "--no-auto", "--no-shell", f"--user={self.user}", \
            *self.OUTPUT_ARGS, self.qube, "--", *cmd]
        track_commandline(args)
        p = subprocess.run(args, capture_output=True, text=True, **kwargs)
        if "check" not in kwargs:
            if p.stderr:
                print_bg_msg(p.stderr)
            p.check_returncode()
        return p

    def run_bool_shellcmd(self, shellcmd:str, falsecodes=range(1,125), **kwargs):
        # this func exists because boolean results need to be done with
        # shell commands, because the alternative (qubes.VMExec I think?)
        # doesn't give the real return code (could be a bug?). That would
        # limit our ability in some cases to differentiate an actual "false"
        # result from the command, vs. an error from qvm-run or failure of
        # the vm to call the cmd.
        # p.s.: qvm-run decides which service to call in 
        # https://github.com/QubesOS/qubes-core-admin-client/blob/main/
        # qubesadmin/tools/qvm_run.py in the run_command_single function.
        #
        # falsecodes : return codes which should be considered a "false"
        # response from the shell command. It should never include the
        # following known non-false codes:
        # 0: success, obviously
        # 125: QREXEC_EXIT_PROBLEM "Problem with qrexec itself". Reference:
        #      reference: https://github.com/QubesOS/qubes-core-qrexec/blob/
        #                 96813fdde17e85477ca8fd2e644cbc298053f32d/libqrexec
        #                 /qrexec.h#L182
        # 126: QREXEC_EXIT_REQUEST_REFUSED
        # 127: Command not found error
        # <0 or >127: Process aborted early by a signal

        args = ["qvm-run", "--no-auto", f"--user={self.user}"] \
            + [self.qube, "--", shellcmd]
        track_commandline(args)
        p = subprocess.run(args, stderr=subprocess.PIPE, **kwargs)

        if p.returncode == 0:
            return True
        regex = rb"(\n|^)qvm-run\W+error"
        if p.returncode not in falsecodes or re.search(regex, p.stderr):
            print_bg_msg(p.stderr.decode())
            p.check_returncode()
        return False

    def dataset_exists(self, dataset):
        return self.run_bool_shellcmd("zfs list "+dataset)

    def get_space_avail(self):
        args = "zfs list -H -p -o avail".split() + [self.dest_dataset]
        p = self.run_cmd(*args)
        return int(p.stdout)

    @staticmethod
    def new_transfer_metrics():
        return {"bytes_sent": 0, "secs_elapsed": 0}

    async def stream_to(self, local_cmd, recv_cmd, metrics=None,
                        progress_cb:Callable[[dict], None] = (lambda x: None)):
        if not all(
            (isinstance(c, collections.abc.Sequence) and not isinstance(c, str))
            for c in (local_cmd, recv_cmd)
        ): raise TypeError()
        if metrics is None:
            metrics = self.new_transfer_metrics()
        metrics_ref_time = float(metrics['secs_elapsed'])
        enter_time = time.time()

        def update_metrics(sz=0):
            metrics['bytes_sent'] += sz
            metrics['secs_elapsed'] = (time.time() - enter_time) + metrics_ref_time
            progress_cb(metrics)
            return metrics

        async def metrics_update_on_interval():
            interval = 0.2
            # reference time is offset by half of interval to avoid having
            # metrics['secs_elapsed'] land too close to whole numbers (which could
            # give the appearance of a buggy clock if progress callback is
            # used to print whole seconds).
            ref_time = enter_time - (metrics_ref_time-int(metrics_ref_time)) + (interval/2)
            t = enter_time
            while True:
                # Interval alignment is always based on ref_time. But if somehow
                # an entire interval is missed, we don't try to immediately
                # iterate twice to "catch up"-- the second iter simply doesn't happen.
                await asyncio.sleep(((ref_time - t) % interval) + 0.000001)
                t = time.time()
                metrics['secs_elapsed'] = (t - enter_time) + metrics_ref_time
                progress_cb(metrics)

        read_limit = 1024 ** 2
        local_proc, recv_proc = await asyncio.gather(
            asyncio.create_subprocess_exec(
                *local_cmd,
                stdout=asyncio.subprocess.PIPE,
                limit=(2 * read_limit),
            ),
            asyncio.create_subprocess_exec(
                "/usr/bin/qvm-run", "--no-auto", "--no-shell", f"--user={self.user}",
                *self.OUTPUT_ARGS, self.qube, "--", *recv_cmd,
                stdin=asyncio.subprocess.PIPE,
            )
        )

        metrics_idle_task = asyncio.create_task(metrics_update_on_interval())
        out = b''
        while True:
            update_metrics(len(out))
            out = await local_proc.stdout.read(read_limit)
            await recv_proc.stdin.drain()
            if not out:
                break
            recv_proc.stdin.write(out)

        metrics_idle_task.cancel()
        recv_proc.stdin.close()
        await recv_proc.stdin.wait_closed()
        await asyncio.wait(
            (asyncio.create_task(local_proc.wait()),
            asyncio.create_task(recv_proc.wait()),
            metrics_idle_task)
        )

        return update_metrics()

def zfs_ancestors_iter(ds):
    while True:
        ds,sep,_ = ds.rpartition('/')
        if not sep:
            return
        yield ds

def qubesd_query(call, dest, arg=None):
    # usage reference: https://www.qubes-os.org/doc/admin-api/
    args = ["qubesd-query","-e","dom0",call,dest,*([arg] if arg is not None else [])]
    track_commandline(args)
    p = subprocess.run(args, capture_output=True)
    ret_code = p.stdout[:2]
    if ret_code != b'0\0':
        err = ' '.join([repr(ret_code), "from qubesd-query", call, dest, arg or ''])
        raise MyError(err, error_code="fatal")
    text_output = p.stdout[2:].decode("ascii")
    lines = list(filter(bool, text_output.splitlines()))
    try:
        return {k:v for k,v in (l.split('=',maxsplit=1) for l in lines)}
    except ValueError:
        return lines

class LazySystemQuery():
    def __init__(self, args, postprocess=(lambda x: x), cmd=None):
        self.args = args
        self.postprocess = postprocess
        self.cmd = cmd or run_cmd
        self.cache = None

    def get(self, refresh_cache=False):
        if refresh_cache or (self.cache is None):
            self.cache = self.postprocess(self.cmd(*self.args))
        return self.cache

    @staticmethod
    def get_zfs_qubes_pools_info():
        pools = qubesd_query("admin.pool.List","dom0")
        r = {}
        for p in pools:
            pinfo = qubesd_query("admin.pool.Info","dom0",p)
            if pinfo["driver"] == "zfs":
                r[p] = pinfo
        return r

qube_list = LazySystemQuery(["qvm-ls","--raw-list"], (lambda x: set(str.split(x))))
dataset_list = LazySystemQuery("zfs list -H -o name".split(), (lambda x: set(str.splitlines(x))))
qubes_pools_info = LazySystemQuery([], cmd=LazySystemQuery.get_zfs_qubes_pools_info)

def contained_zvols_attached_to_any_domain(dataset):
    """The specified dataset and all of its descendants are are checked
    to see if they are attached to any qubes, and if so, the return value
    is the set of those datasets which are attached. If none of the datasets
    are attached to a qube, an empty set is returned. Volumes permanently
    attached to a NON-running qube don't count as attached for this-- at
    least, they shouldn't, and if they do, it's a bug (I haven't tested)."""
    devnodes = set()
    for dom in qapp.domains:
        get_attached_blockdevs = \
            getattr(dom.devices['block'], "get_attached_devices", None) or \
            dom.devices['block'].attached
        for blk in get_attached_blockdevs():
            devnodes.add(blk.data['device_node'])
    all_attached_zvols = set()
    for devnode in devnodes:
        try:
            all_attached_zvols.add(run_cmd("/lib/udev/zvol_id",devnode).strip())
        except subprocess.CalledProcessError:
            pass

    args = f"zfs list -H -o name -t volume -r {dataset}".split()
    contained_zvols = set(run_cmd(*args).split())
    return contained_zvols & all_attached_zvols

class SubcommandsList():
    def __init__(self):
        self.list_ = []
        self.by_name = {}

    def __iter__(self):
        return iter(self.list_)

    def by_tags(self, *tags):
        if not tags:
            return
        tags = set(tags)
        for cmd in self:
            if tags <= cmd["tags"]:
                yield cmd

    def add(self, f, tags):
        name = f.__name__.lstrip('_')
        spec = {
            "cmd_id": len(self.list_),
            "name": name,
            "help": f.__doc__,
            "tags": set(tags),
            "func": f,
        }
        assert name not in self.by_name
        self.list_.append(spec)
        self.by_name[name] = spec

    def get_add_decorator(self, *tags):
        def d(f):
            self.add(f, tags)
            return f
        return d

subcommands = SubcommandsList()
subcmd = subcommands.get_add_decorator

def get_cli_options():
    def error(msg):
        options['verbose'] = False
        raise MyError(msg, error_code='input')

    if len(sys.argv) == 1:
        error(f"Use '{apparent_command_name} help' for help.")
    elif sys.argv[1:] in (["-h"],["--help"]):
        cmd_name = 'help'
    else:
        cmd_name = sys.argv[1]
        if cmd_name.startswith('-'):
            error("No command specified")
        elif cmd_name not in subcommands.by_name:
            error(f"Command '{cmd_name}' doesn't exist.")

    cmd = subcommands.by_name[cmd_name]

    p = argparse.ArgumentParser(prog=f"{apparent_command_name} {cmd['name']}")
    p.add_argument('-v', dest="verbose", action="store_true")
    if 'config_name' in cmd['tags']:
        p.add_argument("config_name")
    if 'file1' in cmd['tags']:
        p.add_argument("file1")
    opts = p.parse_args(sys.argv[2:])

    return cmd, dict(vars(opts))



###################################################################

@subcmd('backupcmd','file1')
def _check():
    """\
    Performs validation on a new user-provided configuration prior to importing.
    The user should study the output carefully to catch any configuration
    mistakes, and recheck if revisions are made.
    """
    config = Config.import_(options["file1"])
    config.print_import_analysis()
    if config.name in Config.all_config_names():
        warn(f"A configuration named {config.name} already exists, " \
            "so importing with that name will fail.")
    else:
        print("Configuration can be imported")
    if config.user_config['receiver']['qube'] not in qapp.domains:
        warn(f"Receiving qube {config.user_config['receiver']['qube']} does not " \
            "exist and will need to be created before a send can be initiated.")

@subcmd('backupcmd','file1')
def _import():
    """\
    Imports the user's configuration into the program's internal store.
    Configurations must be imported before they can be operated on
    (snapshotting, sending, etc.).
    """
    if not os.path.isdir(options["metadata_dir"]):
        e = FileNotFoundError(options["metadata_dir"])
        msg = "options.metadata_dir is not valid; you may " \
            "need to create the directory"
        raise MyError(e, msg, error_code='fatal')

    config = Config.import_(options["file1"])
    if config.name in Config.all_config_names():
        msg = "Refusing to overwrite configuration of same name"
        raise MyError(msg, error_code='fatal')
    config.save()
    print("Imported "+config.name)

@subcmd('backupcmd', 'config_name')
def _snapshot():
    """\
    Creates the configuration's target snapshot.
    """
    config = Config.load(options["config_name"])
    try:
        config.do_snapshots()
    finally:
        config.save()

@subcmd('backupcmd', 'config_name')
def _calc():
    """\
    Calculates size of the eventual send, and performs some final checks before
    the send.
    """
    config = Config.load(options["config_name"])
    try:
        config.calculate_send()
    finally:
        config.save()

@subcmd('backupcmd', 'config_name')
def _send():
    """\
    Copies the included set to the receiver qube. If it succeeds, the backup
    process is complete.
    """
    config = Config.load(options["config_name"])
    try:
        config.do_send()
    finally:
        config.save()

@subcmd('infocmd')
def _list():
    """\
    Lists all imported configurations
    """
    for c in Config.list_configs():
        print(*c)

@subcmd('infocmd')
def _help():
    """Prints command list"""
    from textwrap import dedent
    indent = ' '*4
    vspace = ''

    # define command groups
    groups = [
        {
            "cmds": [],
            "head": "Backup user commands:\n"+indent+"To perform a backup, "
                    "each of these should be called sequentially.",
            "member_test": (lambda c: 'backupcmd' in c['tags']),
            "number": True,
            "show": "yes",
        },
        {
            "cmds": [],
            "head": "Informational user commands:",
            "member_test": (lambda c: 'infocmd' in c['tags']),
            "number": False,
            "show": "yes",
        },
        {
            "cmds": [],
            "head": "Non-user commands:",
            "member_test": (lambda c: True),
            "number": False,
            "show": "verbose",
        },
    ]
    for cmd in subcommands:
        for g in groups:
            if g['member_test'](cmd):
                g['cmds'].append(cmd)
                break

    # print help
    print()
    print(f"Usage: {apparent_command_name} <command> [-hv] [<options>...]\n")
    description = PROGRAM_DESCRIPTION[0].upper() + PROGRAM_DESCRIPTION[1:].rstrip('.')
    print(indent + description + ".\n")

    print("Global options:" + vspace)
    print(indent + "-h           show command-specific help information")
    print(indent + "-v           verbose mode")
    print()

    for g in groups:
        if g['show'] == 'no' or (g['show'] == 'verbose' and not options['verbose']):
            continue

        print(g['head'] + vspace)
        for i,cmd in enumerate(g['cmds'], start=1):
            enum = (str(i)+'. ') if g['number'] else ''
            name = colours.BOLD + cmd['name'] + colours.RESET
            padding = ' ' * max(0,8-len(cmd['name']))
            doc = ' '.join(dedent(cmd['help'] or '?').split('\n'))

            print(f"{indent}{enum}{name}{padding} - {doc}{vspace}")
        print()

    options['verbose'] = False

@subcmd()
def _dev_stream_test():
    options['verbose'] = True
    r = Receiver("backups-disp",Receiver.NULL_DATASET,user="user")
    # print(r.dataset_exists("heeee"))
    print("hi")
    x = r.stream_to(["/root/backup_test/lil-outputter"],
        ["md5sum"], progress_cb=lambda s: print(s["secs_elapsed"]))
    asyncio.run(x)

def main():
    try:
        cmd, o = get_cli_options()
        options.update(o)
        cmd["func"]()
    except MyError as e:
        print(e, file=sys.stderr)
        sys.exit(e.error_code)
    except Exception as e:
        traceback.print_exception(e)
        sys.exit(ERROR_CODES["fatal"])
    else:
        match warning_counter:
            case 0:
                w_text = "with 0 warnings."
            case 1:
                w_text = f"but with {colours.WARN}1 warning{colours.RESET}."
            case w:
                w_text = f"but with {colours.WARN}{w} warnings{colours.RESET}."
        if warning_counter or options["verbose"]:
            print("Program finished successfully,",w_text, file=sys.stderr)

if __name__ == "__main__":
    main()
