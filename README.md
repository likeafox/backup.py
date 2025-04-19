backup.py is a backup and snapshotting tool for QubesOS+ZFS systems.

## Project Goals

### Minimize user error in configuring backups

Misconfiguration of backups is particularly concerning because it can easily go undetected until it has already lead to data loss. While some user error in configuring backups is inevitable, it can be greatly reduced with a program that gives feedback on the user's choices.

backup.py provides information to the user as comprehensively as possible, in situations where that information can be useful in finding and correcting accidental omissions or other mistakes.

### Be trusted to run in dom0

There is no small barrier to using 3rd party software inside dom0, because it requires ultimate trust in that software. The need to ultimately trust my backup program helped motivate me to write my own, and trust is essential if the program is to be useful to anyone else.

backup.py attempts to be auditable by keeping the features and dependencies to the minimum necessary to reach its other primary goals.

I also intend to make signed releases available in the future.

### Support QubesOS/ZFS-specific integration and features

backup.py makes the backup process paradigmatic to QubesOS and ZFS. It a) isolates the writing of backup media within a dedicated qube, and b) uses zfs-native send/receive commands to replicate or amend (as in incremental backups) datasets.

It also avoids pitfalls that general-purpose backup programs can run into when they lack awareness of QubesOS or ZFS. For example, a program needs awareness of volumes in-use by running qubes to avoid snapshotting volumes in an inconsistent state.

Presently, backup.py will only back up datasets, but not a qube's settings. So while it's useful for preserving data, it is not exactly helpful with regard to fully-restoring a functioning Qubes system. This is a limitation that may be addressed in the future.

### Facilitate multitasking workflow

When the backup process would temporarily interfere with other things that need to be done, such as doing work or powering down the system, it is common for users to postpone the backup instead. That impacts the timeliness and usefulness of backups.

To minimize those types of inconveniences, backup.py divides backups into smaller units of work that need not all be completed at once. A backup has many points that it can be interrupted and resumed from, and much of it can also be done in parallel with other tasks.

## Installation

#### Basic installation

1) Copy backup.py to wherever you will run it from (it will be run by root@dom0-- simply putting it in `/root/exampledir` is fine). You should be familiar with [how to copy to dom0](https://www.qubes-os.org/doc/how-to-copy-from-dom0/#copying-to-dom0).
2) In the same directory that contains backup.py, create a directory named `metadata`. This is where backup.py will store information about your backups.

You can now start using it! Run `python3 /path/to/backup.py help` to see the list of commands.

#### Optional steps

- Just as you should for any foreign file going into dom0, I suggest checking the integrity of backup.py before use. Either compare to a known hash, or open in `nano` and inspect the code to make sure there are no surprises.
- For easier use, make it into a standalone command:
```bash
chmod u+x /path/to/backup.py
ln -s /path/to/backup.py /root/.local/bin/backup.py
backup.py help
```
- If you want, you can change the location of the metadata directory. It is in the list of options near the top of the backup.py file.

## Typical Workflow

#### Step 1: Writing the config file

Everything about a backup is defined up-front, in the user's backup configuration file. The reason for this is so we can catch as many potential mistakes as possible, before any modification makes it to disk. Therefore, every backup job begins with its own config file. Most of the time, you won't be writing the config file from scratch, but will copy the config of the previous backup you did, and make slight modifications. The config file format is described in a later section.

#### Step 2: Validating the config

You should always validate the config using the `check` command, as this is where most feedback happens. Carefully review it for the lists of included and excluded datasets, and any warnings or errors. If anything looks wrong, go back and edit the config, and then recheck until you're sure it's correct.

#### Step 3: Creating snapshots

The first action you perform against a new backup configuration is `snapshot`. When you run `snapshot`, the program will try to either create the snapshot for each dataset being backed up, or verify that the snapshot already exists. Only datasets not in use are able to be snapshotted, but you don't need to shut down all your qubes at once; the snapshotting can be staggered. And if you want to truly minimize the downtime of a certain qube, invoke it as one swift series of calls like so: `backup.sh snapshot mybackup1 ; qvm-shutdown --wait work && { sleep 1 ; backup.sh snapshot mybackup1 ; qvm-start work ; }`

#### Step 4: Activating the backup media

The only criterion of a backup media is for it to be a zpool running in an app qube, standalone qube, or disposable qube. Since how that happens is left up to the user, there are more than a few ways to achieve it. Here's a basic case:

1) Plug in your external usb hard drive, and sys-usb receives the device.
2) From the devices menu, find the backup partition on that drive and attach it to the backup qube.
3) In the backup qube import the zpool of your attached block device, ex. `zpool import backuppool`
4) If the dataset that will receive the backup doesn't exist yet, create it: `zfs create -o volmode=none backuppool/backups/mydesktop`

#### Step 5: Sending the backup

Finally, you send your backup to the receiving backup media. This part is the least involved in terms of user interaction, but it is also the part that can take a very, very long time. If needed, you can interrupt a backup at any time with ctrl-C. The next time you start the send it will resume from the last dataset it was working on.

## Backup Config File Format

A user's backup configuration file defines everything about a backup job.

- The file is interpreted as YAML. You may use any standard YAML syntax in your config.
- Most fields are required (we like to be as explicit as possible in defining our backups, and not assume what a default value is).

### Example config file

_backup1.yaml_

```yaml
name: mybackup1

scope:
  target-snapshot: "@backup-2025-04-14"
  since: beginning
  progressive: no
  honour-origins: no

allowed-behaviours:
  snapshot-creation: yes
  patching: no

receiver:
  qube: backups-disp
  dataset: backuppool/backups/mydesktop

include-groups:
  - core-qubes
  - dom0
  - misc
forgo-groups:
  - system
void-groups: []


# group definitions

core-qubes:
  type: qubes
  members:
    - social
    - personal-email
    - financial-accts
    - vault
    - work
    - work-email

misc:
  type: qubes
  members:
    - video-editing

dom0:
  type: datasets
  members:
    - mainpool/dom0

system:
  type: qubes
  members:
    - fedora-41-xfce
    - fedora-41-minimal
    - debian-12-xfce
    - default-dvm
    - default-mgmt-dvm
    - sys-firewall
    - sys-net
    - sys-usb
```

### Variables

#### name

_(string)_

While the filename is how the config is referred to _before_ importing, `name` is how the config is referred to _after_ importing. It can be any string that is able to be recieved as a commandline argument.

You may wish to make `name` the same as the target snapshot label, to make it easy to remember the relationship between your backup's config and the snapshots on your system. In that case, just be aware that can get confusing if you backup the same snapshot to multiple destinations, since the name will have to change each time you back it up, so only one backup will retain the same name as the snapshot label.

#### scope

Scope variables define which parts of any included datasets will be backed up.

#### scope.target-snapshot

_(snapshot-label)_

Name of the snapshot that will always be included in the backup. It is also the _last_ (in terms of chronological creation) snapshot to be included in the backup.

Any included dataset's target snapshot that doesn't exist yet will be created by the `snapshot` command, if snapshot creation is permitted with `allowed-behaviours.snapshot-creation`.

A _snapshot-label_ always begins with an `@` symbol. Example: `"@backup-2025-04-14"`

#### scope.since

_(snapshot-label) | beginning_

This is the start point of a backup. The backup will include data/snapshots _chronologically-after_ `since` up until the target snapshot.

If you have previously made backups to the receiver, set `since` to be the target snapshot of your previous backup. If you haven't made a backup yet, set it to the special value `beginning`.

In ZFS terms, setting `since` to `beginning` is equivalent to doing a _full send_, and setting it to a _snapshot-label_ is equivalent to doing an _incremental send_.

#### scope.progressive

_(boolean)_

Specifies whether the backup should include intermediate snapshots that might exist between `since` and `target-snapshot`. If this is set to `no`, only the target snapshot will be sent to the receiver (with a few exceptions such as satisfying origin dependencies).

#### scope.honour-origins

_(boolean)_

Specifies whether ZFS dataset origins will ever be preserved as part of the backup. `no` means origins won't be preserved. If set to `yes`, backup.py will prefer to preserve origins but can still silently decide to not do so in some cases, such as if an origin dataset is designated _void_.

Disabling origins can reduce disk usage in the narrow case that an origin dataset is not included in the backup, **and** it is only an origin to one other dataset.

#### allowed-behaviours

Some program behaviours are able to be disabled with the aim of being more explicit with configuration. Leaving them all enabled will not interfere with your backup, but you may miss seeing important errors that could inform you there is a problem with your configuration. It's best to only enable the functionality you know your backup needs.

For your first backup, you probably want `snapshot-creation: yes` and `patching: no`.

#### allowed-behaviours.snapshot-creation

_(boolean)_

You may disable this if:

- all missing target snapshots should be created manually, or
- you are repeating a backup to a different backup media and the target snapshots were already created by your previous backup.

#### allowed-behaviours.patching

_(boolean)_

Controls whether the receiver should be expected to potentially already contain some of the backup targets, in which case the backup will behave like a patch, only sending the targets which are missing on the receiver.

#### receiver

Receiver variables specify the destination of the backup

#### receiver.qube

_(string)_

Name of the qube the backup will be sent to. The qube should be an _app_, _disposable_, or _standalone_ qube with qrexec and ZFS capabilities. There is otherwise no specific requirement for the qube itself. The qube may be referred to as the _backup qube_, the _receiver qube_, or simply the _receiver_.

#### receiver.dataset

_(zfs-dataset)_

ZFS dataset in the receiver qube that will contain the completed backup. The dataset must exist.

Going by the example config from earlier, consider the backup of the dataset named `mainpool/qubes/work-email/private`. Once backed up, your backup media will contain the corresponding snapshot named `backuppool/backups/mydesktop/mainpool/qubes/work-email/private@backup-2025-04-14`.

#### include-groups

_(list)_

List of object groups that will have the _include_ designation applied. Objects with the _include_ designation will be included in the backup.

To understand how designations interact, see the later topic on _backup target_ resolution.

#### forgo-groups

_(list)_

List of object groups that will have the _forgo_ designation applied. The _forgo_ designation  indicates the user has declined the object's inclusion in the backup, but it may still be included in special cases. This is also the default for datasets that have not been given a designation, explicitly nor implicitly.

When you want to exclude an object from the backup, _forgo_ is the preferred designation in most cases.

#### void-groups

_(list)_

List of object groups that will have the _void_ designation applied. Objects with the _void_ designation are hard-excluded from the backup. Datasets resolved from these objects are **never** directly included, but note that a partial send of their underlying data blocks may occur if those data blocks are shared by an included dataset.

The _void_ designation should be used to exclude objects:

- When the object represents a dataset that is the child/descendant of an _included_ dataset, but you still want to exclude it in spite of that.
- As a form of redaction, keeping in mind the above caveat.

In most cases, forgo-groups should be used instead.

### Object Groups

These are arbitrarily-named top-level items in your config file. They represent lists of objects on your system-- qubes and ZFS datasets-- that will be given designations to be either included or excluded from the backup. You should think of each of their objects as also implying their descendant ZFS datasets (so the dataset `mainpool/shared` also implies `mainpool/shared/apps` and so on). For a fuller explanation of the semantic, see the next section.

You can organize objects into as many or as few groups as you like.

Example object group:

```yaml
core-qubes:
  type: qubes
  members:
    - social
    - personal-email
    - financial-accts
    - vault
    - work
    - work-email
```

In the example, "core-qubes" is the name given to the group. The name can be anything and is only used to refer to the group where listed under one of `include-groups`, `forgo-groups`, or `void-groups`.

Its `type` field must be either `qubes` or `datasets`.

`members` is a list of qube names (as _list(string)_) in the case of `type: qubes`; or, a list of dataset names (as _list(zfs-dataset)_) in the case of `type: datasets`.

Object Groups as an organizational tool exists so that we can minimize user error in cases where whole groups of objects need to be managed together. Take the following situation:

Your system uses about 1200GB of disk space. You want to back up everything and you have multiple backup media, but none of them can quite fit the entire 1200GB. What you can do is create two main groups for inclusion, of about 600GB each, and then backup only one at a time to each backup media. You might call the groups `qubes-group-1` and `qubes-group-2`. For the first backup, you'll have `qubes-group-1` under `include-groups`, and `qubes-group-2` under `forgo-groups`. For the second backup, you'll swap their places. The benefit here is 1) there is no need to modify the groups themselves, and 2) you have easy and immediate confirmation that they have exactly swapped roles, without mistakes. If the same thing had been done without groups, and involved dozens of objects, you wouldn't be able to tell that at a glance.

### Backup object to backup target resolution

_The following in-depth explanation is useful to know about when moving beyond the most basic use cases._

In the config, each listed qube or dataset is more broadly referred to as an _object_. The objects are a higher-level/abstract way to specify what is to be backed up. When you import a configuration, objects go through a few transformations.

Firstly, each _object_ is transformed to one or more _designations_. Designations are what you see in the output of the `check` command.

In the case of transforming a dataset object to a designation, there is actually no change (object `mainpool/shared` becomes designation `mainpool/shared`). However, in the case of a qube, the qube name is resolved to its root dataset. And although it is extremely uncommon, a qube can have multiple root datasets in multiple Qubes pools, all of which become designations. For example the qube object `games` can become designations `mainpool/qubes/games` and `auxpool/qubes/games` if games's private volume is in mainpool but its root volume is in auxpool.

Designations also have a _type_, which is one of `include`, `forgo`, or `void`. Types are assigned by the inclusion in one of the (include/forgo/void) `*-groups` lists.

Nesting designations of type `include` or `forgo` under other designations is not allowed (you cannot include `mainpool/shared` but forgo `mainpool/shared/apps`). By contrast `void` designations can exist anywhere.

The second transformation is _include designation_ &rarr; _backup targets_. Backup targets are what actually get backed up, and are in the form of a fully-qualified snapshot. You see backup targets when you run the `calc` command.

In the most common case, a designation and each of its child/decendant datasets become the basis of the backup targets, with `scope.target-snapshot` providing the snapshot label. For example:

```
  |  (qube-object from a config file)
  v
work
  |
  |  (transform to include-designation)
  v
mainpool/qubes/work
  |
  |  (transform to backup targets)
  v
mainpool/qubes/work@backup-2025-01-19
mainpool/qubes/work/private@backup-2025-01-19
mainpool/qubes/work/root@backup-2025-01-19
```

Some of the other logics that go into the _include designation_ &rarr; _backup targets_ transformation:

- Transient qube volumes are omitted (the "volatile" volume being the obvious example)
- Datasets within `void` designations are omitted
- Additional targets may be included when they are required for a progressive send
- If permitted, origin snapshots are attempted to be resolved and included
