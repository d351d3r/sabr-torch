# SABR Torch Cpp

### Install torch
```shell
$ wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
$ unzip libtorch-shared-with-deps-latest.zip
```
### Install
```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=./libtorch ..
$ cmake --build . --config Release
```
### Run: 
```shell
$ ./sabr-torch
```


### Build in Mac M1 with [Lima](https://github.com/lima-vm/lima)

#### Install Lima first with brew:
```shell
$ brew install lima
```
You can use your own Lima configuration file or use the author's

For better compatibility, it is recommended to use the architecture: `arch: x86_64`

Please note that the author's configs from the container have full access to the file system for convenience. Instead, you can give access to certain directories:

```yaml
mounts:
- location: "~/project/"
  mountPoint: "/project"
  writable: true
```

#### Run Lima with author's config:
```bash 
$ limactl start --name=default etc/lima-default.yaml
```

#### Go to the working directory via Lima:

```shell
$ lima
```
#### Install dependencies:
```shell 
$ sudo apt install -y gcc g++ make cmake
````


#### Build:

```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=./libtorch ..
$ cmake --build . --config Release

```
#### Run: 
```shell
$ ./sabr-torch
```
