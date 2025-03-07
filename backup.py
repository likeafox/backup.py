#!/usr/bin/python3

options = {
    "metadata_dir": "/home/likeafox/backup_test/metadata",
    # verbose is here so it can have a value at program-initialization time
    # gets overwritten by cli opts early on
    "verbose": True
}
ERROR_CODES = {
    "input": 1,
    "fatal": 2,# all fully-unhandled errors go here
    "yaml": 3,
    "notfound": 4,# only when specifying non-existant files on the command line
}

import sys, traceback, itertools, os.path
import argparse, re, subprocess, datetime, json
import types, collections.abc
#from copy import deepcopy

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

warning_counter = 0
def warn(msg):
    global warning_counter
    warning_counter += 1
    print(f"{colours.WARN}Warning: {msg}{colours.RESET}", file=sys.stderr)

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
    error_code = ERROR_CODES["input"]

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
        "ver": "1",
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
        h_sz = sum(next(zip(defin)))
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
    PERSISTENT = ('user_config','raw_user_config', 'objects')

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
        graph = cls._make_graph(objects)

        o = cls()
        o.graph = graph
        vars(o).update((k,v) for k,v in locals().copy().items() if k in cls.PERSISTENT)
        o.header = ConfigHeader()
        o.header["importdate"] = \
            datetime.datetime.now().isoformat(sep=' ',timespec='minutes')

        o._serialize()
        print(f"Config validation {colours.OK}PASSED{colours.RESET}")
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
        return os.path.join(options["metadata_dir"], name + ".backupcfg")

    def print_import_analysis(self):
        print("Analysis:")
        self._print_coverage_analysis(self.graph)
        self._print_snapshots_analysis(self.user_config, self.objects)

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
                o["datasets"] = [next(iter(vol_parents))]
            else:
                o["datasets"] = [i["vid"] for i in infos]
        elif o['type'] == "datasets":
            o["name"] = "dataset:" + m
            o["datasets"] = [m]
        else:
            assert False
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
            for i,ds in enumerate(o["datasets"]):
                node = graph[ds]
                if node["own_obj"] is not None:
                    msg = f"Duplicate reference conflicts with {node['own_obj'][0]['name']}"
                    errs += o["name"], msg
                # parents = [(pds,graph[pds]) for pds in parents_iter(ds)]
                ancestor_objs = filter(bool, (graph[pds]["own_obj"] for pds in zfs_ancestors_iter(ds)))
                ancestor_collisions = ((a,(o,i)) for a in ancestor_objs)
                descendant_collisions = (((o,i),d) for d in node["descendant_objects"])
                for a,d in itertools.chain(ancestor_collisions, descendant_collisions):
                    errs += a[0]["name"], d[0]["name"]+" is nested in "+a[0]["datasets"][a[1]]

                node["own_obj"] = (o,i)
                for pds in zfs_ancestors_iter(ds):
                    graph[pds]["descendant_objects"].append((o,i))

            for ids in o["ignore_datasets"]:
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

    @staticmethod
    def _print_snapshots_analysis(user_config, objects):
        # if snapshot creation is disabled, check if snapshots exist already (and warn if not)
        # otherwise, if it's enabled, check if snapshotting will be blocked
        def get_snapshots_for(ds):
            p = subprocess.run(f"zfs list -H -t snapshot -r {ds}".split(),
                capture_output=True, check=True, text=True)
            return p.stdout.splitlines()
        #
        target = user_config["scope"]["target-snapshot"][1:]
        snaps_exist = 0
        snaps_to_create = 0
        snaps_blocked = 0
        for o in objects:
            if o['role'] == 'exclude':
                continue
            for i,ds in enumerate(o["datasets"]):
                has_snap = 'no'
                for snap in get_snapshots_for(ds):
                    sds,sep,label = snap.partition('@')
                    assert sep == '@'
                    if label == target:
                        if sds == ds:
                            has_snap = 'yes'
                            break
                        else:
                            has_snap = 'descendant'
                match has_snap:
                    case 'no':
                        snaps_to_create += 1
                    case 'descendant':
                        # only descendant snapshot(s) exist
                        warn(f"Snapshot of {ds} is blocked and will " \
                            "need to be created manually.")
                        snaps_blocked += 1
                    case 'yes':
                        snaps_exist += 1
        total_snaps = snaps_exist + snaps_to_create + snaps_blocked
        #
        print()
        print(f"{snaps_exist}/{total_snaps} required snapshots currently exist.")
        if user_config["scope"]["allow-snapshot-creation"]:
            print(f"{snaps_to_create} snapshots are able to be automatically created.")
        elif snaps_to_create + snaps_blocked > 0:
            print("scope.allow-snapshot-creation is disabled and so snapshots " \
                "must be created manually.")

    def _serialize(self):
        obj = dict((name, getattr(self,name)) for name in self.PERSISTENT)
        return json.dumps(obj, indent=2).encode('ascii', errors='strict')

def zfs_ancestors_iter(ds):
    while True:
        ds,sep,_ = ds.rpartition('/')
        if not sep:
            return
        yield ds

def qubesd_query(call, dest, arg=None):
    args = ["qubesd-query","-e","dom0",call,dest,*([arg] if arg is not None else [])]
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
        if cmd is not None:
            self.cmd = cmd
        self.cache = None

    @staticmethod
    def cmd(*args):
        if options["verbose"]:
            cmdline = ' '.join([(f'"{a}"' if ' ' in a else a) for a in args])
            print(f"{colours.BRIGHT_BLACK}{cmdline}{colours.RESET}", file=sys.stderr)
        p = subprocess.run(args, capture_output=True, check=True, text=True)
        return p.stdout

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
    cmd = subcommands.by_name[sys.argv[1]]

    p = argparse.ArgumentParser(prog=cmd["name"])
    p.add_argument('-v', dest="verbose", action="store_true")
    if 'file1' in cmd["tags"]:
        p.add_argument("file1")
    opts = p.parse_args(sys.argv[2:])

    return cmd, dict(vars(opts))



###################

@subcmd('file1')
def _check():
    config = Config.import_(options["file1"])
    config.print_import_analysis()

@subcmd('file1')
def _import():
    if not os.path.isdir(options["metadata_dir"]):
        e = FileNotFoundError(options["metadata_dir"])
        msg = "options.metadata_dir is not valid; you probably " \
            "need to create the directory"
        raise MyError(e, msg, error_code='fatal')

    config = Config.import_(options["file1"])
    config.save()

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
