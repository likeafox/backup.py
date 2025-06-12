#!/usr/bin/python3

# likeafox's backup and snapshots tool for QubesOS+ZFS systems
# (c) 2025 Jason Forbes <contact@jasonforbes.ca>

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

options = {
    # metadata_dir is where imported configs are stored. It can be an
    # absolute path, or a path relative to this program's location
    "metadata_dir": "metadata",
    # verbose is here so it can have a value at program-initialization time;
    # gets overwritten by cli opts early on
    "verbose": True,
    # enables printing debug messages
    "debug": True,
}

VERSION = "0.1.1"
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

# libraries reference:
# qubesadmin:
# https://dev.qubes-os.org/projects/core-admin-client/en/latest/py-modindex.html
# libzfs_core:
# https://pyzfs.readthedocs.io/en/latest/index.html
#
import sys, traceback, itertools, os, os.path, time, \
    argparse, re, subprocess, signal, datetime, json
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

def compose(*funcs):
    return lambda x: functools.reduce(lambda v,f:f(v), funcs, x)

apparent_command_name = os.path.basename(sys.argv[0])
real_command_dir, real_command_name = \
    os.path.split(os.path.realpath(sys.argv[0], strict=True))
if not options['metadata_dir'].startswith('/'):
    options['metadata_dir'] = os.path.join(real_command_dir, options['metadata_dir'])



warning_counter = 0
warned_once = set()
def warn(msg, prefix="Warning: ", count=1, once=False):
    global warning_counter
    if once:
        if msg in warned_once:
            return
        warned_once.add(msg)
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
            return False
        try:
            self.cur_item = next(self.its)
        except StopIteration as e:
            self.stop = e
        else:
            self.cur_it = iter(self.cur_item)
        return self.stop is None

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

    def __init__(self, verrs=None):
        self.verrs = []
        if verrs is not None:
            for e in verrs:
                self += e

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
    PERSISTENT = ('user_config', 'raw_user_config', 'objects', 'targets')

    @classmethod
    def import_(cls, raw_user_config):
        user_config = cls._parse_user_config(raw_user_config)
        cls._check_objs_exist(user_config)
        objects = [cls._mk_objects_pt2(o) for o in cls._mk_objects_pt1_gen(user_config)]

        o = cls()
        vars(o).update((k,v) for k,v in locals().copy().items() if k in cls.PERSISTENT)
        o.header = ConfigHeader()
        o.header["importdate"] = \
            datetime.datetime.now().isoformat(sep=' ',timespec='minutes')

        o._resolve_targets(o.get_designation_graph())
        o._serialize()
        for target in o.targets:
            o._check_on_snapshot_state(target)
        print(f"Config validation {colours.OK}PASSED{colours.RESET}")
        time.sleep(0.9)
        return o

    @classmethod
    def import_file(cls, filename):
        try:
            with open(filename) as f:
                raw_user_config = f.read()
        except FileNotFoundError as e:
            err_str = f"{e.strerror}: {e.filename}"
            raise MyError(err_str, error_code='notfound')

        return cls.import_(raw_user_config)

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

    def get_designation_graph(self):
        try:
            r = self._des_graph
        except:
            r = self._des_graph = self._make_designation_graph(self.objects)
        return r

    @functools.cached_property
    def objects_by_name(self):
        return {o['name'] : o for o in self.objects}

    def print_import_analysis(self):
        print("Analysis:")
        self._print_coverage_analysis(self.get_designation_graph())
        print()
        self._print_snapshots_analysis()

    def do_snapshots(self):
        deferred = 0
        creation_allowed = self.user_config['allowed-behaviours']['snapshot-creation']
        for target in self.targets:
            if target['snap_creation'] in ('unknown','blocked'):
                self._check_on_snapshot_state(target)

            # can we snap?
            if not creation_allowed:
                continue
            if target['snap_creation'] != 'would':
                continue
            ds = target['snapshot'][0]
            attachment = get_zvols_attached_to_any_qube().get(ds)
            if attachment and attachment['rw']:
                hosts = ", ".join(attachment['hosts'])
                warn(f"Volume {ds} is in use by {hosts}. Deferring snapshot creation.")
                deferred += 1
                continue

            # yes, snap
            snap_name = '@'.join(target['snapshot'])
            run_cmd("zfs","snapshot",snap_name)
            target['snap_creation'] = 'done'
            print("snapshotted "+ds)

        self._print_snapshots_analysis()
        if deferred:
            warn(f"{deferred} deferred snapshots. Re-run the command once " \
                "the preconditions are satisfied.", prefix='', count=0)

    @functools.cache
    def get_receiver(self, user):
        return Receiver(**self.user_config['receiver'], user=user)

    def calculate_send(self):
        calc_count = 0
        unsnapped_count = 0

        for target in self.targets:
            if type(target['send_size']) is int:
                continue
            if target['snap_creation'] not in ('done','yes','ignore'):
                unsnapped_count += 1
                continue

            out = run_cmd(*self._make_send_cmd(target,'calc'))
            for l in out.splitlines():
                fields = l.split()
                if fields[:1] == ["size"]:
                    sz = int(fields[1])
                    break
            else:
                raise Exception("size not found")
            target['send_size'] = sz
            target['send_status'] = 'ready'
            calc_count += 1

        if calc_count:
            print(f"Calculated size of {calc_count} objects")
        if unsnapped_count:
            warn(f"{unsnapped_count} snapshots are missing. Calculations for " \
                "those objects have been skipped.")
        self._print_calc_result(True)

    def do_send(self, *, phases=('interrupted','depth'), complete_send_only=True):
        total_bytes = 0
        initial_sent_bytes = 0
        for t in self.targets:
            if t['send_status'] != 'notready':
                total_bytes += t['send_size']
                if t['send_status'] == 'done':
                    initial_sent_bytes += t['send_size']
            elif complete_send_only:
                raise Exception("can't send: previous phases are incomplete")

        targets_graph = TargetDependencyGraph(self.targets)
        targets_iter = \
            AdvanceableChainIter(*(targets_graph.sendables_iter(ph) for ph in phases))
        exceptions = []
        receiver = Receiver(**self.user_config['receiver'])
        transfer_metrics = receiver.new_transfer_metrics()
        transfer_metrics['bytes_sent'] = initial_sent_bytes
        progress = ProgressBar(total_bytes)

        # don't exit haphazardly if the program is interrupted (set signal handlers)
        signame = None
        def get_interrupted(sig, _):
            nonlocal signame
            signame = {2: "SIGINT", 15: "SIGTERM"}.get(sig, "(unknown signal)")
            print(f"\ngot {signame}. exiting soon", file=sys.stderr)
            while targets_iter.advance():
                pass
        other_sig_handlers = {
            signal.SIGINT: signal.signal(signal.SIGINT, get_interrupted),
            signal.SIGTERM: signal.signal(signal.SIGTERM, get_interrupted),
        }

        # main send loop
        for t in targets_iter:
            ds, snaplabel = t['snapshot']
            local_snap = f"{ds}@{snaplabel}"
            dest_ds = f"{receiver.dest_dataset}/{ds}"
            dest_snap = f"{dest_ds}@{snaplabel}"
            dest_parent = dest_snap.rsplit('/', maxsplit=1)[0]
            incr_src = t['incremental_source']
            try:
                if receiver.dataset_exists(dest_snap):
                    if self.user_config['allowed-behaviours']['patching']:
                        total_bytes -= t['send_size']
                        t['send_size'] = 0
                        t['send_status'] = 'done'
                        progress = ProgressBar(total_bytes)
                        print(local_snap,"already exists on receiver. Marking as complete.")
                        continue
                    else:
                        raise ReceiverError(f"Target {local_snap} already exists at destination")

                dest_ds_exists = receiver.dataset_exists(dest_ds)
                if not dest_ds_exists:
                    if incr_src and incr_src[0] == ds:
                        raise ReceiverError("incremental source doesn't exist on receiving end.")
                    # create parent container if it doesn't exist
                    receiver.run_cmd(*("zfs create -p -u".split()), dest_parent)
                    if t['send_status'] == 'ready':
                        # This is the first attempt sending (otherwise send_status would be
                        # 'interrupted'), and dest_ds doesn't exist so this rollback option
                        # can be set
                        t['rollback_option'] = 'delete-ds'

                if t['send_status'] == 'interrupted' and options['allow-rollback']:
                    # follow-up attempt; try rollback
                    if t['rollback_option'] == 'delete-ds':
                        if dest_ds_exists:
                            receiver.run_cmd("zfs", "destroy", "-r", dest_ds)
                            receiver.run_cmd(*f"zfs wait -t deleteq {dest_parent}".split())
                    elif incr_src and incr_src[0] == ds:
                        warn("Rollback of incremental send is not implemented. " \
                            "If a rollback proves necessary, it will have to be " \
                            "done manually.", once=True)

                local_cmd = self._make_send_cmd(t, 'send')
                recv_cmd = ["zfs", "receive"]
                if options['zfs-overwrite']:
                    recv_cmd += ["-F"]
                if incr_src and incr_src[0] != ds:
                    recv_cmd += [
                        "-o",
                        f"origin={receiver.dest_dataset}/{incr_src[0]}@{incr_src[1]}",
                    ]
                recv_cmd += [dest_ds]

                # send! HOORAY!
                print("sending", local_snap)
                send = receiver.stream_to(
                    local_cmd, recv_cmd, transfer_metrics, progress.print_cb
                )
                asyncio.run(send)

                time.sleep(0.2)
                if options['verbose']:
                    print()
                if receiver.dataset_exists(dest_snap):
                    # Done!
                    if options['verbose']:
                        print("\nOK")
                    else:
                        progress.backspace(True)
                    t['send_status'] = 'done'
                else:
                    # :(
                    raise ReceiverError("Unable to confirm existence of newly sent snapshot")

            except Exception as e:
                print(f"{colours.ERR}{e}{colours.RESET}", file=sys.stderr)
                exceptions += [f"error while attempting to send {local_snap}", e]
                t['send_status'] = 'interrupted'
                #
                targets_iter.advance()

        # we're done with the heavy stuff; revert signal handlers
        for x in other_sig_handlers.items():
            signal.signal(*x)

        # report on how things went
        sent = transfer_metrics['bytes_sent'] - initial_sent_bytes
        if sent:
            print("sent " + human_readable_bytesize(sent).strip(' -'))
        if exceptions:
            print(f"{colours.ERR}send failed{colours.RESET}", file=sys.stderr)
            raise MyError(*exceptions, error_code='send')
        elif signame:
            warn("send exited early!")
        else:
            remain = len(targets_graph)
            if remain:
                print(f"Stopped send with {remain} unsent targets remaining.")
            else:
                print("send completed successfully!")

    @staticmethod
    def _designating_groups(user_config):
        for cat in ("include","forgo","void"):
            for gn in user_config[cat+"-groups"]:
                yield cat,gn

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
        test("scope.honour-origins", bool)
        test("allowed-behaviours.snapshot-creation", bool)
        test("allowed-behaviours.patching", bool)
        test("receiver.qube", str, cat="word")
        test("receiver.dataset", str, cat="dataset")

        if all(test(l+"-groups", list) for l in ("include","forgo","void")):
            dgs = list(Config._designating_groups(config))
            groups_set = set(gn for cat,gn in dgs)
            group_types = ("qubes","datasets")
            for g in groups_set:
                if test(g, cat="group"):
                    test(g+".type", validator=lambda x: x in group_types)
                    test(g+".members", list)
            if len(dgs) != len(groups_set):
                validation_errors += "groups", "Includes/forgos/voids contain redundant group references"

        if validation_errors:
            raise validation_errors
        #else
        return config

    @staticmethod
    def _check_objs_exist(user_config):
        """check if all referenced qubes and datasets exist
        (except for receiver.dataset and all things in unused groups)"""
        def all_of(t):
            for c,gn in Config._designating_groups(user_config):
                if user_config[gn]["type"] == t:
                    yield from user_config[gn]["members"]
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
        """atomize include/forgo/void designators into standalone objects,
        and resolve designated datasets"""
        for cat,gn in Config._designating_groups(user_config):
            for m in user_config[gn]["members"]:
                yield {
                    "role_designation": cat,
                    "groupname": gn,
                    "member_name": m,
                    "type": user_config[gn]["type"],
                }

    @staticmethod
    def _mk_objects_pt2(o):
        m = o["member_name"]

        if o['type'] == 'qubes':
            o['name'] = "qube:" + m
            vols = qubesd_query("admin.vm.volume.List",m)
            vol_parents = {}
            infos_to_incl = []
            vids = set()
            for vol in vols:
                if vol == "kernel":
                    continue
                vol_info = qubesd_query("admin.vm.volume.Info",m,vol)
                if vol_info['ephemeral'] != "False":
                    raise Exception(["idk what ephemeral does",m,vol,vol_info])
                if vol_info['pool'] not in qubes_pools_info.get():
                    warn(f"not all of qube {m}'s volumes are in "
                        "an applicable pool. Some volumes won't be included")
                    continue
                vids.add(vol_info['vid'])
                vol_parent, vol_name = vol_info['vid'].rsplit('/',maxsplit=1)
                assert vol == vol_name
                if vol_info['save_on_stop'] != "True":
                    vol_parents.setdefault(vol_parent, False)
                    continue
                vol_parents[vol_parent] = True
                infos_to_incl.append(vol_info)

            o['datasets'] = [ds for ds,incl in vol_parents.items() if incl] + \
                            [i['vid'] for i in infos_to_incl]
            o['ignore_datasets'] = [ds for ds,incl in vol_parents.items() if incl == False]

            # warn for missed non-qube datasets
            prefixes = [k+'/' for k in vol_parents]
            for ds in dataset_list.get():
                for pref in prefixes:
                    if ds.startswith(pref) and ds not in vids:
                        warn(f"{ds} does not belong to qube {m}, and won't be included.")
                        break

        elif o['type'] == 'datasets':
            o['name'] = "dataset:" + m
            args = "zfs list -H -t filesystem,volume -o name -r".split() + [m]
            o['datasets'] = run_cmd(*args).splitlines()
            o['ignore_datasets'] = []
        else:
            assert False
        return o

    @staticmethod
    def _make_designation_graph(objects):
        """make graph, checking for duplicate and/or nested (in zfs dataset hierarchy) designations"""
        @(lambda x: dict(x()))
        def graph():
            qubes_containers = '|'.join(pi["container"] for pi in qubes_pools_info.get().values())
            disp_pat = re.compile('(' + qubes_containers + r')/disp\d{1,4}')
            for ds in dataset_list.get():
                k = tuple(ds.split('/'))
                node = {
                    "k": k,
                    "ds": ds,
                    "ancestors": [k[:n] for n in range(len(k)-1, 0, -1)],
                    "des_objs": [], #objects designating this node
                    "descendant_des": [], #descendant designations
                    "ignore": (bool(disp_pat.fullmatch(ds)) or '/.' in ds),
                }
                yield k,node

        def designations_of(obj):
            dataset_ks = set(tuple(ds.split('/')) for ds in o['datasets'])
            for k in dataset_ks.copy():
                if dataset_ks & set(graph[k]['ancestors']):
                    dataset_ks.remove(k)
            return dataset_ks
        #
        errs = ConfigError()
        def collision_err(anc_ds, anc_o, desc_ds, desc_o):
            if anc_ds == desc_ds:
                if anc_o['name'] == desc_o['name']:
                    msg = "Duplicate objects"
                else:
                    msg = f"Designation on {'/'.join(anc_ds)} conflicts with {desc_o['name']}"
            else:
                msg = f"{desc_o['name']} is nested in {'/'.join(anc_ds)}"
            errs += anc_o['name'], msg
        #
        for o in objects:
            for des in designations_of(o):
                node = graph[des]
                for other in node['des_objs']:
                    # direct collision
                    collision_err(des, o, des, other)
                node['des_objs'].append(o)

                for desc in node['descendant_des']:
                    for other in graph[desc]['des_objs']:
                        if other['role_designation'] in ('include','forgo'):
                            # collision by occluding other
                            collision_err(des, o, desc, other)

                for anc in node['ancestors']:
                    anc_node = graph[anc]
                    anc_node['descendant_des'].append(des)
                    for other in anc_node['des_objs']:
                        if o['role_designation'] in ('include','forgo'):
                            # collision by nesting in other
                            collision_err(anc, other, des, o)

            for ids_str in o["ignore_datasets"]:
                ids = tuple(ids_str.split('/'))
                if ids in graph:
                    graph[ids]["ignore"] = True
        if errs:
            raise errs

        for node in graph.values():
            node['ancestor_des_exists'] = False
            relevant_nodes = [node] + [graph[a] for a in node['ancestors']]
            for other_n in relevant_nodes:
                if other_n['des_objs']:
                    [other_obj] = other_n['des_objs']
                    if other_n is not node:
                        node['ancestor_des_exists'] = True
                    if node['ds'] in other_obj['datasets']:
                        node['role_obj'] = other_obj
                        node['role'] = other_obj['role_designation']
                        break
            else:
                node['role_obj'] = None
                node['role'] = 'forgo'
        return graph

    def _resolve_targets(self, des_graph):
        assert not hasattr(self, "targets")
        progressive = self.user_config['scope']['progressive']
        target_snaplabel = self.user_config['scope']['target-snapshot'].removeprefix('@')
        since_snaplabel = (lambda x: None if x == 'beginning' else x.removeprefix('@')) \
                          (self.user_config['scope']['since'])
        snap_txg_map = {snap : txg for txg,snap in snap_txg_order.get()}
        max_snap_txg = snap_txg_order.get()[-1][0]
        snap_txg_order_by_ds = collections.defaultdict(list)
        prev_txg = -1
        for txg,(ds,label) in snap_txg_order.get():
            assert txg >= prev_txg
            snap_txg_order_by_ds[ds].append((txg,label))
            prev_txg = txg

        prototargets = {} # dict of (target : incr_src)
        targets_to_add = {
            (n['ds'], target_snaplabel) for n in des_graph.values() \
                if n['role'] == 'include'
        }
        snapshottable_targets = targets_to_add.copy()
        while targets_to_add:
            ds,snaplabel = targets_to_add.pop()
            all_snaps = []
            for txg,label2 in snap_txg_order_by_ds[ds]:
                all_snaps.append(label2)
                if label2 == snaplabel:
                    break

            incr_src = None # as in "incremental source" from zfs-send man page
            if since_snaplabel in all_snaps:
                incr_src = (ds,since_snaplabel)
            elif self.user_config['scope']['honour-origins']:
                orig_map = origin_map.get()
                if ds in orig_map:
                    orig_ds, orig_snaplabel = orig_map[ds]
                    can_incl_orig = des_graph[tuple(orig_ds.split('/'))]['role'] != 'void'
                    if can_incl_orig or orig_snaplabel == since_snaplabel:
                        incr_src = (orig_ds, orig_snaplabel)

            prototargets[(ds,snaplabel)] = incr_src
            if incr_src is None:
                # there's no origin or we're ignoring the origin.
                if since_snaplabel is not None:
                    warn(f"Not using 'since' snapshot for {ds}. Will do a full send.")
                if progressive and all_snaps:
                    # a primer is needed because zfs-send isn't otherwise able to send
                    # a progressive full stream without the -R option. Thus the primer
                    # is sent as a regular full stream of the very first snap, and then
                    # send -I of the following target snap completes the progressive send.
                    prototargets[(ds,all_snaps[0])] = None
            else:
                if incr_src[1] != since_snaplabel:
                    # didn't hit 'since' yet. Keep including deeper origins til we get there
                    assert incr_src[0] != ds
                    targets_to_add.add(incr_src)

        # if there are duplicate datasets, need to link up their snaps properly
        for ds,targets_it in itertools.groupby(sorted(prototargets), lambda x: x[0]):
            this_ds_targets = set(targets_it)
            ordered_snaps = [(ds,l) for txg,l in snap_txg_order_by_ds[ds] \
                            if (ds,l) in this_ds_targets]
            ordered_snaps += list(this_ds_targets - set(ordered_snaps))
            for source,derivative in zip(ordered_snaps[:-1], ordered_snaps[1:]):
                prototargets[derivative] = source

        @(lambda f: dict(f()))
        def user_ordered_objects():
            for i,o in reversed(list(enumerate(self.objects))):
                if o['role_designation'] != 'include':
                    continue
                v = (i,o['name'])
                for ds in o['datasets']:
                    yield ds, v

        self.targets = []
        for snap,incr_src in prototargets.items():
            ds, snaplabel = snap
            hierarchy_k = ds.split('/')
            sortkey_obj, obj_name = user_ordered_objects.get(ds, (len(self.objects)+1,None))
            sortkey_txg = snap_txg_map.get(snap, max_snap_txg+1)
            t = {
                "snapshot": [ds, snaplabel],
                "incremental_source": (incr_src and list(incr_src)),
                "hierarchy_k": hierarchy_k,
                "origin_dep": (bool(incr_src) and incr_src[1] != since_snaplabel),
                "hierarchy_dep": any(ds.startswith(ds2 + '/') for ds2,_ in prototargets),
                "progressive": progressive and bool(incr_src),
                "snap_creation": ("unknown" if snap in snapshottable_targets else "ignore"),
                "send_size": None,
                "send_status": "notready",
                "rollback_option": None,
                "sortkey": [sortkey_obj, hierarchy_k, sortkey_txg],
                "obj_name": obj_name,
            }
            self.targets.append(t)

    @staticmethod
    def _print_coverage_analysis(des_graph):
        # analyze coverage (warn for qubes/datasets neither included nor excluded)
        designations = { 'include': [], 'forgo': [], 'void': [], }
        unspecified = []
        ignores = 0
        irrel_in_des = 0
        irrel_default = 0
        for k,node in sorted(des_graph.items()):
            if node['des_objs']:
                [o] = node['des_objs']
                designations[o['role_designation']].append(node['ds'])
            elif node['ignore']:
                ignores += 1
            elif node['ancestor_des_exists']:
                irrel_in_des += 1
            elif not node['descendant_des'] and (not node['ancestors'] or \
                    des_graph[node['ancestors'][0]]['descendant_des']):
                unspecified.append(node['ds'])
            else:
                irrel_default += 1

        global warning_counter
        warning_counter += len(unspecified)

        print("\nIncluded datasets:")
        posi = f"{colours.GREEN}+{colours.RESET}"
        print(posi,f'\n{posi} '.join(designations['include']))

        print("\nForgone datasets; the user has declined their inclusion, but " \
            "they may still be included in special cases. This is also the " \
            "default for datasets that are not included (explicitly or " \
            "implicitly) in any group:")
        nega = f"{colours.RED}-{colours.RESET}"
        if len(designations['forgo']):
            print(nega,f'\n{nega} '.join(designations['forgo']))
        else:
            print("(none)")

        print("\nVoided (hard-excluded) datasets; these are NEVER directly " \
            "included (but a partial send of their data blocks may occur if " \
            "those data blocks are shared by an included dataset):")
        nega = f"{colours.RED}-{colours.RESET}"
        if len(designations['void']):
            print(nega,f'\n{nega} '.join(designations['void']))
        else:
            print("(none)")

        if len(unspecified) or options['verbose']:
            print(f"\n{colours.BRIGHT_YELLOW}Warning: The following datasets have not been " \
                "referenced. Consider explicitly including or excluding them:")
            print('-','\n- '.join(unspecified),f"{colours.RESET}")

        print(f"\nAdditionally:\n{ignores} datasets have been automatically ignored " \
            f"by the program's built-in rules.\n{irrel_default} datasets have " \
            "been implicitly ignored based on the user's designating groups.\n" \
            f"{irrel_in_des} are not being shown because they are descendant " \
            "datasets belonging to designations already shown above.")

    def _check_on_snapshot_state(self, target, reset=False):
        if target['snap_creation'] == 'ignore' \
            or (target['snap_creation'] in ('yes','done') and not reset):
            return
        assert target['snap_creation'] in ('unknown','yes','done','would','blocked')
        has_snap = dataset_exists('@'.join(target['snapshot']))
        target['snap_creation'] = {False: 'would', True: 'yes'}[has_snap]

    def _print_snapshots_analysis(self):
        count = collections.Counter()
        for target in self.targets:
            count[target['snap_creation']] += 1
        if count['unknown'] > 0:
            if count.total() == (count['unknown'] + count['ignore']):
                warn("Requested snapshots analysis results but no analysis has been done")
            else:
                warn("Refusing to print incomplete snapshot analysis results. " \
                    f"({count['unknown']} objects would be omitted)")
            return

        total = count.total() - count['ignore']
        print(f"{count['yes']+count['done']}/{total} required snapshots currently exist.")
        if self.user_config['allowed-behaviours']['snapshot-creation']:
            print(f"{count['would']} snapshots are able to be automatically created.")
        elif count['would'] + count['blocked'] > 0:
            print("allowed-behaviours.snapshot-creation is disabled and so snapshots " \
                "must be created manually.")
        if count['unknown'] > 0:
            warn(f"Incomplete snapshot analysis: {count['unknown']} objects were omitted.")

    def _serialize(self):
        obj = dict((name, getattr(self,name)) for name in self.PERSISTENT)
        return json.dumps(obj, indent=2).encode('ascii', errors='strict')

    @staticmethod
    def _make_send_cmd(target, mode):
        c = ["zfs", "send"]
        c += {'calc':["--dryrun","-P"], 'send':[]}[mode]
        c += ["-p"]
        if target['incremental_source'] is not None:
            c += [("-i","-I")[target['progressive']]]
            c += ['@'.join(target['incremental_source'])]
        c += ['@'.join(target['snapshot'])]
        return c

    def _print_calc_result(self, query_receiver=False):
        total = 0
        skipped = 0
        size_results = []
        for target in self.targets:
            sz = target['send_size']
            if type(sz) is int:
                total += sz
                sk = target['sortkey']
                name = '@'.join(target['snapshot'])
                size_display = human_readable_bytesize(sz)
                size_results.append( [sk, name, size_display] )
            else:
                skipped += 1
        if not size_results:
            warn("calc results are empty")
            return
        size_results.sort()

        print("\nsend size:")
        for _,name,size in size_results:
            print(size,name)

        if not skipped:
            print(human_readable_bytesize(total),"(total)")

            if query_receiver:
                avail = self.get_receiver("user").get_space_avail()
                readable_avail = human_readable_bytesize(avail).strip(' -')
                print("Available space on receiver:", readable_avail)
                if (total / avail) > 0.75:
                    ratio = str(round(total / avail * 100)) + "%"
                    warn(f"Send size is {ratio} of available space on receiver. " \
                        "Because of possible differences in pool/dataset " \
                        "configuration (block size, compression type, etc.), " \
                        "this program is unable to predict whether sent data " \
                        "will fit in the available space. Until technology " \
                        "improves, that determination is left up to the user. ")
        else:
            warn("There are uncalculated targets. Results are incomplete.")

class TargetDependencyGraph():
    def __init__(self, targets):
        self.targets_map = {tuple(t['snapshot']) : t for t in targets}
        sgs = {type_ : self._mk_subgraph(targets, type_) for type_ in ('origin','hierarchy')}
        self.subgraphs = sgs
        self.origin_graph = sgs['origin']
        self.hierarchy_graph = sgs['hierarchy']

    def __len__(self):
        lens = []
        for g in (self.origin_graph, self.hierarchy_graph):
            self._update(g)
            count = 0
            for count,_ in enumerate(self._walk_subgraph(g), start=1):
                pass
            lens.append(count)
        assert lens[0] == lens[1]
        return lens[0]

    def sendables(self):
        def keys_of(g):
            self._update(g)
            return {tuple(t['snapshot']) for t,c in g}
        shared_keys = keys_of(self.origin_graph) & keys_of(self.hierarchy_graph)
        if len(shared_keys) == 0 and len(self) != 0:
            if options['debug']:
                self._print_subgraph(self.origin_graph, "origin")
                print("")
                self._print_subgraph(self.hierarchy_graph, "hierarchy")
            raise RuntimeError("Graph iteration has deadlocked")
        r = [self.targets_map[k] for k in shared_keys]
        r.sort(key=(lambda t: t['sortkey']))
        return r

    def sendables_iter(self, mode):
        def by_status(s):
            return filter((lambda t: t['send_status'] == s), self.sendables())

        def depth():
            while True:
                try:
                    yield next(iter(by_status('ready')))
                except StopIteration:
                    return

        def breadth():
            while True:
                row = list(by_status('ready'))
                if not row:
                    return
                yield from row

        modes = {
            "depth": depth,
            "breadth": breadth,
            "interrupted": (lambda: by_status('interrupted')),
            "one": (lambda: self.sendables()[:1]),
        }

        return iter(modes[mode]())

    @staticmethod
    def _mk_subgraph(targets, type_):
        g = [] # graph with nodes of (target,[children...])
        to_add = []
        for t in targets:
            # create all nodes
            # nodes without dependency are placed directly in g as root nodes
            {True: to_add, False: g}[t[type_+"_dep"]].append((t,[]))

        # organize to_add nodes
        if type_ == 'hierarchy':
            for depth in itertools.count(1):
                if not to_add:
                    break
                at_depth = [n for n in to_add if len(n[0]['hierarchy_k']) == depth]
                for adding_n in at_depth:
                    to_add.remove(adding_n)
                    adding_k = adding_n[0]['hierarchy_k']
                    match_len = 0
                    match_n = None
                    for parent_n in TargetDependencyGraph._walk_subgraph(g):
                        parent_k = parent_n[0]['hierarchy_k']
                        cmp_len = min(len(parent_k), len(adding_k)-1)
                        txg_bias = int(match_n is not None and \
                                    parent_n[0]['sortkey'][-1] < match_n[0]['sortkey'][-1])
                        if parent_k == adding_k[:cmp_len] and (cmp_len + txg_bias) > match_len:
                            match_len = cmp_len
                            match_n = parent_n
                    match_n[1].append(adding_n)
        elif type_ == 'origin':
            for _ in range(len(to_add)+1):
                for n in to_add.copy():
                    for orig_n in TargetDependencyGraph._walk_subgraph(g):
                        if orig_n[0]['snapshot'] == n[0]['incremental_source']:
                            orig_n[1].append(n)
                            to_add.remove(n)
                            break
                if not to_add:
                    break
            else:
                raise Exception("incremental source not found")
        return g

    @staticmethod
    def _walk_subgraph(g):
        for n in g:
            yield n
            yield from TargetDependencyGraph._walk_subgraph(n[1])

    @staticmethod
    def _update(g):
        for i,(t,c) in reversed(list(enumerate(g))):
            if t['send_status'] not in ('ready','interrupted'):
                TargetDependencyGraph._update(c)
                g[i:i+1] = c

    @staticmethod
    def _print_subgraph(g, name="subgraph"):
        root_c = [{"snapshot":(name,)},g]
        path = [[root_c,0]]
        while path:
            (t,c),ci = n = path[-1]
            if not ci:
                indent = ' ' * (4 * (len(path) - 1))
                print(indent, '@'.join(t['snapshot']), sep='')
            try:
                path.append([c[ci],0])
                n[1] += 1
            except IndexError:
                path.pop()

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
        progress_cb_cooldown = 1.0

        def update_metrics(sz=0):
            nonlocal progress_cb_cooldown
            t = time.time()
            metrics['bytes_sent'] += sz
            metrics['secs_elapsed'] = (t - enter_time) + metrics_ref_time
            if t >= progress_cb_cooldown:
                progress_cb_cooldown = t + 0.05
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
        full_recv_cmd = ["/usr/bin/qvm-run", "--no-auto", "--no-shell", \
            f"--user={self.user}", *self.OUTPUT_ARGS, self.qube, "--", *recv_cmd,]
        track_commandline([*local_cmd, '|', *full_recv_cmd])
        #
        local_proc, recv_proc = await asyncio.gather(
            asyncio.create_subprocess_exec(
                *local_cmd,
                stdout=asyncio.subprocess.PIPE,
                limit=(2 * read_limit),
            ),
            asyncio.create_subprocess_exec(
                *full_recv_cmd,
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
        for role,proc in [("local",local_proc), ("recv",recv_proc)]:
            if proc.returncode != 0:
                raise ReceiverError(f"{role} proc exited: {proc.returncode}")

        return update_metrics()

class ProgressBar():
    def __init__(self, total_bytes):
        self.total_bytes = total_bytes
        self.total_bytes_str = self.bytes_str(total_bytes)
        self._update_env()
        self.last_content_len = 1

    def backspace(self, flush=False):
        print(' '*(self.last_content_len+1), end='\r')

    def print_cb(self, kwargs):
        now = time.time()
        if now > self.last_update_env + 0.5:
            self._update_env(now)

        content = self.render(**kwargs)
        self.backspace()
        try:
            print(content, end='\r', flush=True)
        except BlockingIOError:
            time.sleep(0.2)
            if options['debug']:
                msg = "ProgressBar.print_cb: print call threw BlockingIOError"
                warn(msg, prefix="Debug: ", once=True)
        self.last_content_len = len(content)

    def render(self, bytes_sent, secs_elapsed):
        ratio = bytes_sent / self.total_bytes
        percent = (str(round(ratio * 100, 1)) + "%").rjust(7)
        bytes_sent_str = self.bytes_str(bytes_sent)
        parts = [
            percent+" ",
            " " + self.time_str(secs_elapsed) + " ",
            f"{bytes_sent_str} / {self.total_bytes_str}".rjust(14),
        ]

        max_bar_segs = self.term_width - (sum(map(len, parts)) + 8)
        if max_bar_segs >= 10:
            bar_segs = round(float(max(0, min(1, ratio))) * max_bar_segs)
            bar = '[' + ('#'*bar_segs).ljust(max_bar_segs) + ']'
        else:
            bar = '|'

        parts[1:1] = [bar]
        content = ''.join(parts)[:self.term_width-2]
        return content

    @staticmethod
    def bytes_str(sz):
        return human_readable_bytesize(sz).rstrip(' -').replace(' ','')

    @staticmethod
    def time_str(secs:int):
        mins, sec_part = divmod(int(secs), 60)
        hrs, min_part = divmod(mins, 60)
        return "{}:{:02}:{:02}".format(hrs, min_part, sec_part)

    def _update_env(self, now=None):
        self.term_width = os.get_terminal_size().columns
        self.last_update_env = now or time.time()

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

    @staticmethod
    @(lambda f: compose(f, dict))
    def post_origin_map(raw):
        for l in raw.splitlines():
            if '@' in l:
                k,v = l.split()
                yield k, tuple(v.split('@'))

    @staticmethod
    @(lambda f: compose(f, list))
    def post_snap_txg(raw):
        for l in raw.splitlines():
            if '@' in l:
                a,b = l.split()
                yield int(a), tuple(b.split('@'))

qube_list = LazySystemQuery(["qvm-ls","--raw-list"], (lambda x: set(str.split(x))))
dataset_list = LazySystemQuery("zfs list -H -o name".split(), (lambda x: set(str.splitlines(x))))
qubes_pools_info = LazySystemQuery([], cmd=LazySystemQuery.get_zfs_qubes_pools_info)
origin_map = LazySystemQuery("zfs list -H -o name,origin".split(), LazySystemQuery.post_origin_map)
snap_txg_order = LazySystemQuery(
    "zfs list -H -t snapshot -o createtxg,name -s createtxg -s name".split(),
    LazySystemQuery.post_snap_txg
)

def dataset_exists(ds):
    args = ["zfs", "list", ds]
    track_commandline(args)
    null = subprocess.DEVNULL
    p = subprocess.run(args, check=False, stdout=null, stderr=null)
    return p.returncode == 0

diskpath_to_dataset_cache = {}
def diskpath_to_dataset(diskpath):
    if not diskpath.startswith("/dev/"):
        return None
    try:
        return diskpath_to_dataset_cache[diskpath]
    except KeyError:
        pass
    real_diskpath = os.path.realpath(diskpath)
    if not os.path.exists(real_diskpath):
        return None
    try:
        zvol = run_cmd("/lib/udev/zvol_id", real_diskpath).strip()
    except subprocess.CalledProcessError:
        zvol = None

    diskpath_to_dataset_cache[diskpath] = zvol
    return zvol

def get_zvols_attached_to_any_qube():
    """\
    Get info on all zvols that are currently attached to any running qube.
    Return value is a mapping where the keys are all the attached (in-use) zvols.
    """
    all_attached_zvols = {}
    domains_info_raw = run_cmd("/usr/sbin/xl", "list", "-l")
    domains_info = json.loads(domains_info_raw)
    for dom in domains_info:
        id_ = dom['domid']
        name = dom['config']['c_info']['name']
        if (id_ == 0) != (name == "Domain-0"):
            raise RuntimeError(f"is this dom0 or isn't it? (id={id_}, name={name})")
        if id_ == 0:
            continue
        for disk_info in dom['config'].get('disks', []):
            zvol = diskpath_to_dataset(disk_info['pdev_path'])
            if zvol is not None:
                zvol_info = all_attached_zvols.setdefault(zvol, {"rw":False, "hosts":[]})
                zvol_info['hosts'].append(name)
                if bool(disk_info.get('readwrite')):
                    zvol_info['rw'] = True
    return all_attached_zvols

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
        def with_config():
            config = Config.load(options["config_name"])
            try:
                f(config)
            finally:
                config.save()
        name = f.__name__.lstrip('_')
        func = with_config if ('with_config' in tags) else f
        spec = {
            "cmd_id": len(self.list_),
            "name": name,
            "help": f.__doc__,
            "tags": set(tags),
            "func": func,
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
    if 'with_config' in cmd['tags'] or 'config_name' in cmd['tags']:
        p.add_argument("config_name")
    if 'send-opts' in cmd['tags']:
        p.add_argument('-f', dest="allow-rollback", action="store_true")
        p.add_argument('-F', dest="zfs-overwrite", action="store_true")
    if 'depgraph' in cmd['tags']:
        p.add_argument('-i','--initial', dest="depgraph-initial", action="store_true")
    if 'file1' in cmd['tags']:
        p.add_argument("file1")

    opts = dict(vars(
        p.parse_args(sys.argv[2:])
    ))

    # extra validation
    if 'send-opts' in cmd['tags'] and opts['zfs-overwrite'] and not opts['allow-rollback']:
        raise argparse.ArgumentError("-F requires -f to also be specified.")

    return cmd, opts



###################################################################

@subcmd('backupcmd','file1')
def _check():
    """\
    Performs validation on a new user-provided configuration prior to importing.
    The user should study the output carefully to catch any configuration
    mistakes, and recheck if revisions are made.
    """
    config = Config.import_file(options["file1"])
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

    config = Config.import_file(options["file1"])
    if config.name in Config.all_config_names():
        msg = "Refusing to overwrite configuration of same name"
        raise MyError(msg, error_code='fatal')
    config.save()
    print("Imported "+config.name)

@subcmd('backupcmd', 'with_config')
def _snapshot(config):
    """\
    Creates the configuration's target snapshot.
    """
    config.do_snapshots()

@subcmd('backupcmd', 'with_config')
def _calc(config):
    """\
    Calculates size of the eventual send, and performs some final checks before
    the send.
    """
    config.calculate_send()

@subcmd('backupcmd', 'with_config', 'send-opts')
def _send(config):
    """\
    Copies the included set to the receiver qube. If it succeeds, the backup
    process is complete.
    """
    config.do_send()

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

@subcmd('config_name')
def _reimport():
    old_config = Config.load(options['config_name'])
    new_config = Config.import_(old_config.raw_user_config)
    new_config.save()
    print(f"Reimported '{new_config.name}' and its old savestate has been overwritten.")

@subcmd('with_config', 'send-opts')
def _send_one(config):
    """\
    Like send, but exits after sending one target.
    """
    config.do_send(phases=('one',))

@subcmd('config_name','depgraph')
def _depgraph():
    config = Config.load(options['config_name'])
    graph = TargetDependencyGraph(config.targets)
    for name,sg in graph.subgraphs.items():
        print()
        if not options['depgraph-initial']:
            graph._update(sg)
        graph._print_subgraph(sg, name)

@subcmd()
def _dev_stream_test():
    options['verbose'] = True
    r = Receiver("backups-disp",Receiver.NULL_DATASET,user="user")
    # print(r.dataset_exists("heeee"))
    print("hi")
    x = r.stream_to(["/root/backup_test/lil-outputter"],
        ["md5sum"], progress_cb=lambda s: print(s["secs_elapsed"]))
    asyncio.run(x)

@subcmd()
def _progress_test():
    start = time.time()
    total = 3_300_000_000
    progress = ProgressBar(total)
    for x in range(0, total, 1337970):
        time.sleep(0.1)
        arg = {'bytes_sent': x, 'secs_elapsed': int(time.time() - start)}
        progress.print_cb(arg)

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
