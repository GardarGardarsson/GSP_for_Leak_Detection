### Connect to GPU Servers:

`ssh -X ucee<initials>@<server>.ee.ucl.ac.uk`

### Mount a `Remote` Directory `Locally` on your Mac OS X 

Install a `FUSE` (**F**ilesystem in **USE**rspace ) program for Mac OS X, like [macFUSE](https://osxfuse.github.io/)

Install `sshfs` e.g. by download here: [sshfs](https://github.com/osxfuse/sshfs/releases/download/osxfuse-sshfs-2.5.0/sshfs-2.5.0.pkg) 

Run:

`sshfs ucee<initials>@<server>.ee.ucl.ac.uk:/<remote>/<directory>/<to>/<mount>/ '/<local>/<mounting>/<directory>'`

**Best practice is to have `/<local>/<mounting>/<directory>` an EMPTY DIRECTORY just for the remote directory, as you will lose access to all the data in the assigned directory while the remote drive is mounted! If you haven't done this, don't panic, your data is still there, just unmount the remote drive, e.g. by dragging the drive icon to the trash **

Now you can start a `ssh` session with the GPU server, e.g. `geneva.ee.ucl.ac.uk` , and create a file in the shared remote directory, e.g. with: `touch test.txt`

You should see that file pop-up in a while on your `local` mount point of the `remote` directory.



