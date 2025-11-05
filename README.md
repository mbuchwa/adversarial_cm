# ad-ldm

This repository contains experiments combining diffusion and consistency models.  
For quick "lightscale" tests, a compact configuration is available at
`config/lightscale_example.yaml`.  It uses a reduced model, a limited dataset
and low epoch counts so you can iterate rapidly while validating ideas.

### Running a lightscale experiment

```
python joint_training.py --config config/lightscale_example.yaml
```

The `GeometriesDataModule` now accepts a `max_samples` argument to restrict the
number of examples loaded from disk.  The provided lightscale configuration sets
`dataset_params.max_samples` to 100.

### Training the CCDM baseline

The CCDM Lightning module follows the same workflow as the joint and CcGAN
trainers.  Launch training with the dedicated entry point and the new
configuration file:

```
python ccdm_training.py --config config/ccdm_geometries.yaml --seed 42
```

The YAML file groups dataset parameters, CCDM model hyperparameters (UNet,
diffusion, label embedding), vicinal weighting options, logging cadence and
sampling behaviour so experiments can be reproduced or tweaked with minimal
edits.

### Training the CcGAN-AVAR variant

The AVAR lightning module reuses the geometry data module and logging helpers.
Start an experiment with the dedicated launcher and configuration template:

```
python ccgan_avar_training.py --config config/ccgan_avar_geometries.yaml --seed 42
```

The AVAR configuration exposes the dataset, label embedding, convolutional
network family, vicinal sampling hyperparameters, auxiliary losses, optimiser
settings and logging cadence so AVAR runs can be reproduced in the same way as
the base CcGAN trainer.  The new `config/ccgan_avar_config.py` dataclasses mirror
the Lightning module arguments and deserialize YAML files into strongly typed
blocks:

- **`dataset`** – forwarded to `GeometriesDataModule`, providing the image root,
  conditioning metadata and batching strategy used when the module is
  instantiated.
- **`label_embedding`** – consumed by the `LabelEmbed` helper to create the
  callable passed to `LightningCcGANAVAR(fn_y2h=...)`.
- **`model`** – defines the backbone (`net_name`), latent dimensions and channel
  multipliers required by `load_network_constructors`.  Flags
  `use_aux_reg_branch`/`use_aux_reg_model` and
  `aux_reg_model_checkpoint` enable either the discriminator branch or a
  checkpointed regression network that is injected through the
  `aux_reg_model` argument.
- **`vicinal`** – mapped directly to the module's `vicinal_params`.  Besides the
  original `kappa` and `threshold_type`, additional switches such as
  `use_ada_vic`, `ada_vic_type`, `ada_eps`, `min_n_per_vic` and
  `use_symm_vic` control the adaptive window construction implemented inside
  `_make_vicinity`.
- **`aux_loss`** – transformed into the `aux_loss_params` dict for the Lightning
  module.  The weights gate the regression (`weight_*_aux_reg_loss`) and DRE
  penalties (`weight_*_aux_dre_loss`), while the optional
  `aux_reg_checkpoint`/`dre_checkpoint` entries document where the respective
  estimators are stored.
- **`optimisation`** – builds `AVAROptimisationConfig`, which in turn configures
  the Lightning optimisers.  In addition to learning rates, batch sizes and
  gradient accumulation counts, the tuple `betas` is threaded to Adam for both
  generator and discriminator.
- **`training`** – controls runtime features when the module is constructed:
  sampling cadence, mixed precision, DiffAug (`diffaug.enabled` &
  `diffaug.policy`) and EMA tracking (`ema.enabled`, `ema.update_after_step`,
  `ema.update_every`, `ema.decay`).
- **`logging`** – initialises the WandB logger and `ModelCheckpoint` callback,
  matching the Lightning trainer configuration.

The default `config/ccgan_avar_geometries.yaml` captures geometry-dataset
settings for quick experiments and can be used as a template for custom runs.

### Launching hyperparameter sweeps

Weights & Biases sweeps are now available for each baseline so you can explore
model and optimiser settings without editing the YAML files manually.  Each
helper mirrors the CLI of the corresponding training entry point:

```bash
python joint_sweep.py --config config/geometries_cond.yaml --project MyProject
python ccgan_sweep.py --config config/ccgan_geometries.yaml --project MyProject
python ccdm_sweep.py --config config/ccdm_geometries.yaml --project MyProject
python ccgan_avar_sweep.py --config config/ccgan_avar_geometries.yaml --project MyProject
```

Use `--method` to switch between `bayes`, `grid` and `random` search strategies
and `--count` to limit the number of runs launched by the sweep agent.  The
helpers clone the base configuration, apply the sampled parameters and dispatch
the training routine so the underlying YAML templates remain unchanged.

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://emcl-gitlab.iwr.uni-heidelberg.de/marcus.buchwald/ad-ldm.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://emcl-gitlab.iwr.uni-heidelberg.de/marcus.buchwald/ad-ldm/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
