# Cyclotron

Cyclotron is the functional RISC-V SIMT simulator of the Muon cores in the Radiance GPU, driven by the gluon command processor.

## Usage

### Building

```bash
cargo build --release
```

### Basic execution

```bash
cargo run --release -- config.toml
```

Or with a specific binary:

```bash
cargo run --release -- config.toml --binary-path path/to/kernel.elf
```

### Common CLI options

| Option | Description |
|--------|-------------|
| `--binary-path <path>` | Override the ELF binary to run |
| `--num-lanes <N>` | Override lanes per warp (default: 16) |
| `--num-warps <N>` | Override warps per core (default: 4) |
| `--num-cores <N>` | Override cores per cluster (default: 1) |
| `--log <level>` | Log level: 0=none, 1=info, 2=debug |
| `--gen-trace <bool>` | Generate instruction trace |

### Example: Run ISA tests

```bash
./test/run-isa-tests.sh
```
