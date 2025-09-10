## About

skvaider is a proxy and router for (Open)AI-APIs.

Features:

* route requests to different backend servers based on model availability and load
* automated health checks
* automated model discovery
* token authentication

> The name `skvaider` is derived from the Swedish word `skvader` which is one
> of those mythical beasts concocted from multiple real animals. Incidentally
> a flying rabbit seems more than apt as this project was started by the
> Flying Circus.

## Development environment

> This requires a local Nix installation.

This will start the development server, postgresql, and other auxiliary services.

```bash
$ nix develop --impure
$ devenv up
```

### Variations of the development environment

You can also activate the development environment with direnv and a `.envrc` (not checked in) using `use flake . --impure`.

```bash
$ nix develop --impure
```

## Testing

```bash
$ nix develop --impure
$ run-tests
```
