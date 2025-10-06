# seg-gui

[![License MIT](https://img.shields.io/pypi/l/seg-gui.svg?color=green)](https://github.com/Ananya-Sundararajan/seg-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/seg-gui.svg?color=green)](https://pypi.org/project/seg-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/seg-gui.svg?color=green)](https://python.org)
[![tests](https://github.com/Ananya-Sundararajan/seg-gui/workflows/tests/badge.svg)](https://github.com/Ananya-Sundararajan/seg-gui/actions)
[![codecov](https://codecov.io/gh/Ananya-Sundararajan/seg-gui/branch/main/graph/badge.svg)](https://codecov.io/gh/Ananya-Sundararajan/seg-gui)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/seg-gui)](https://napari-hub.org/plugins/seg-gui)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A plugin that can be used to better validate and edit segmentation.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template] (None).

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `seg-gui` via [pip]:

```
pip install seg-gui
```

If napari is not already installed, you can install `seg-gui` with napari and Qt via:

```
pip install "seg-gui[all]"
```


To install latest development version :

```
pip install "git+https://github.com/Ananya-Sundararajan/seg-gui.git#egg=seg-gui[all]"
or
pip install git+https://github.com/Ananya-Sundararajan/seg-gui.git
```

## To Use
1. Data Formatting
    Your images and masks should be in separate folders named "images" and "masks" respectively. The images and masks folder should then be in the same directory with no other folders/files in this directory.

2. Loading the data.
    Open napari and click the Seg Gui plugin. Once the seg-gui plugin is open, you should be able to drag the directory that the images and masks folders are in to the seg_gui docker (not the viewer in the middle, but the right-hand panel) to load. Alternatively, you can selet the directory through the load folder button at the bottom of the docker.

3. Validating the data.
    You should be able to hide all masks, delete a specific mask, toggle masks individually on and off, toggle layers on and off, and scroll using 'next' and 'previous' buttons. To check a masks validity in neighboring slices, you can hide the current image layers and show the previous image or the next image.

    After toggling image layers, make sure the mask layer is selected so that you can continue adding masks. You can add masks using napari's layer controls (either paintbrush or polygon work well for our purposes).

    ** When adding masks, the most important thing to note is that you must hit the refresh mask button for the system to trigger the saving of the added masks to the original stack and to the list on the panel (was the cleaner way to ensure new masks were always saved).

4. Saving the masks.
    Select the mask_stack layer, navigate to "Files," and export by clicking "save selected layers."
