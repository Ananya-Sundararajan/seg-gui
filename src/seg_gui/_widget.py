from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class SegmentationEditor(QWidget):

    # initialize with all attributes and widget layout
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.current_idx = 0
        self.images_stack = None  # all image data (array form) in order
        self.masks_stack = (
            None  # all mask data (array fotm), same order as images
        )
        self.images_stack_layer = None
        self.masks_stack_layer = None
        self.middle_mask_layer = None
        self.prev_image_layer = None
        self.middle_image_layer = None
        self.next_image_layer = None
        self.current_mask_ids = []

        self.setAcceptDrops(True)  # for drag and drop

        """ Buttons """
        self.prevBtn = QPushButton("< Previous")
        self.prevBtn.clicked.connect(self.prev_clicked)

        self.nextBtn = QPushButton("Next >")
        self.nextBtn.clicked.connect(self.next_clicked)

        self.loadFolderBtn = QPushButton("Load Folder")
        self.loadFolderBtn.clicked.connect(self.load_folder)

        self.hideAllBtn = QPushButton("Hide All")
        self.hideAllBtn.clicked.connect(self.hide_clicked)

        self.refreshBtn = QPushButton("Refresh Masks")
        self.refreshBtn.clicked.connect(self.on_mask_changed)

        """ Layouts """
        mainLayout = QVBoxLayout()

        scrollLayout = QHBoxLayout()
        scrollLayout.addWidget(self.prevBtn)
        scrollLayout.addWidget(self.nextBtn)

        refLayout = QHBoxLayout()
        refLayout.addWidget(self.refreshBtn)

        togLayout = QHBoxLayout()
        togLayout.addWidget(self.hideAllBtn)

        # masks checkbox scroll area
        self.checkboxArea = QScrollArea()
        self.checkboxContainer = QWidget()
        self.checkboxLayout = QVBoxLayout()
        self.checkboxContainer.setLayout(self.checkboxLayout)
        self.checkboxArea.setWidgetResizable(True)
        self.checkboxArea.setWidget(self.checkboxContainer)

        mainLayout.addLayout(scrollLayout)
        mainLayout.addLayout(refLayout)
        mainLayout.addLayout(togLayout)
        mainLayout.addWidget(self.checkboxArea)
        mainLayout.addWidget(self.loadFolderBtn)

        self.setLayout(mainLayout)

    # load folder (load folder button to set up panels)
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder"
        )
        if not folder:
            return
        from ._reader import reader_function

        images_stack, masks_stack = reader_function(
            folder
        )  # get the data (tuple format)
        self.load_data(
            images_stack[0], masks_stack[0]
        )  # initialize for validation-specific set-up

    # load the data - not actually being called, of no use
    def load_data(self, images_stack, masks_stack):
        self.images_stack = images_stack
        self.images_stack_layer = self.viewer.add_image(
            self.images_stack,  # entire stack
            name="Images Stack",
            visible=False,
        )
        self.masks_stack = masks_stack
        self.masks_stack_layer = self.viewer.add_labels(
            self.masks_stack, name="Masks Stack", visible=False  # entire stack
        )
        self.current_idx = 0

        self.update_panels()  # load in setup

    # detect the drag and drop event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # triggered by the drag and drop
    def dropEvent(self, event):
        folder = event.mimeData().urls()[0].toLocalFile()
        from ._reader import reader_function

        images_stack, masks_stack = reader_function(folder)
        self.load_data(images_stack[0], masks_stack[0])

    # update panels, called when next or previous buttons are clicked
    def update_panels(self):

        # check if there is anything loaded first
        if self.images_stack is None or self.masks_stack is None:
            return

        # define idxes to extract images
        prev_idx = self.current_idx - 1
        next_idx = self.current_idx + 1

        # remove any previous layers
        if (
            self.middle_mask_layer is not None
            and self.middle_mask_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self.middle_mask_layer)
            self.middle_mask_layer = None

        for layer in [
            self.prev_image_layer,
            self.middle_image_layer,
            self.next_image_layer,
        ]:
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)

        # update layers
        self.next_image_layer = (
            self.viewer.add_image(
                self.images_stack[next_idx],
                name="Pos" + str(next_idx) + " - Next",
                visible=True,
            )
            if next_idx <= self.images_stack.shape[0] - 1
            else None
        )

        self.prev_image_layer = (
            self.viewer.add_image(
                self.images_stack[prev_idx],
                name="Pos" + str(prev_idx) + " - Previous",
                visible=True,
            )
            if prev_idx >= 0
            else None
        )

        self.middle_image_layer = self.viewer.add_image(
            self.images_stack[self.current_idx],
            name="Pos" + str(self.current_idx) + " - Current",
        )

        self.middle_mask_layer = self.viewer.add_labels(
            self.masks_stack[self.current_idx].copy(),
            name="Pos" + str(self.current_idx) + " masks",
        )
        self.viewer.layers.move(
            self.viewer.layers.index(self.middle_mask_layer),
            len(self.viewer.layers) - 1,
        )
        self.viewer.layers.selection.active = self.middle_mask_layer

        # Extract IDs for checkboxes
        self.current_mask_ids = np.unique(self.middle_mask_layer.data)
        self.current_mask_ids = self.current_mask_ids[
            self.current_mask_ids != 0
        ]
        self.imageLabel = "Pos" + str(self.current_idx) + " Masks"

        # update glom checkbox view
        self.update_mask_list()

        # for adding
        if self.current_mask_ids is not None:
            next_id = max(self.current_mask_ids) + 1
        else:
            next_id = 1
        self.middle_mask_layer.selected_label = next_id

    # toggle masks when corresponding checkboxes are clicked
    def toggle_mask(self, gid, state):

        if self.middle_mask_layer is None:
            return

        # Convert to a proper ndarray copy
        current_mask = np.array(self.middle_mask_layer.data, copy=True)

        if state:  # checkbox checked
            current_mask[self.masks_stack[self.current_idx] == gid] = gid
        else:  # checkbox unchecked
            current_mask[current_mask == gid] = 0

        # Assign back to trigger redraw
        self.middle_mask_layer.data = current_mask

    # update mask list, called when next or previous buttons are clicked
    def update_mask_list(self):

        # clear old checkboxes/buttons
        for i in reversed(range(self.checkboxLayout.count())):
            layout_item = self.checkboxLayout.itemAt(i)
            if layout_item is not None:
                widget = layout_item.widget()
                if widget:
                    widget.setParent(None)
                else:
                    # It's a layout (HBoxLayout), so we remove its widgets first
                    for j in reversed(range(layout_item.count())):
                        w = layout_item.itemAt(j).widget()
                        if w:
                            w.setParent(None)

        # extract current IDs from middle mask
        mask = self.middle_mask_layer.data
        mask_ids = np.unique(mask)
        mask_ids = mask_ids[mask_ids != 0]
        self.current_mask_ids = mask_ids

        # add checkboxes + delete buttons
        for gid in mask_ids:
            row_layout = QHBoxLayout()

            # Checkbox
            cb = QCheckBox(f"Glom {gid}")
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, g=gid: self.toggle_mask(g, state)
            )
            row_layout.addWidget(cb)

            # Delete button
            btn = QPushButton("Delete")
            btn.clicked.connect(lambda _, g=gid: self.delete_mask(g))
            row_layout.addWidget(btn)

            self.checkboxLayout.addLayout(row_layout)

    # if the mask deletion button is pressed
    def delete_mask(self, gid):
        if self.middle_mask_layer is None:
            return

        # Remove from mask data
        mask_data = np.array(self.middle_mask_layer.data, copy=True)
        mask_data[mask_data == gid] = 0
        self.middle_mask_layer.data = mask_data

        # Remove from the saved stack as well
        self.masks_stack[self.current_idx][
            self.masks_stack[self.current_idx] == gid
        ] = 0

        # Update glom id list
        self.current_mask_ids = [g for g in self.current_mask_ids if g != gid]

        # Remove the corresponding row from the UI
        for i in range(self.checkboxLayout.count()):
            layout_item = self.checkboxLayout.itemAt(i)
            if layout_item is not None:
                for j in range(layout_item.count()):
                    w = layout_item.itemAt(j).widget()
                    if isinstance(w, QCheckBox) and w.text() == f"Glom {gid}":
                        # Remove the entire row layout
                        while layout_item.count():
                            child = layout_item.takeAt(0)
                            if child.widget():
                                child.widget().deleteLater()
                        self.checkboxLayout.removeItem(layout_item)

                        break

    # if the refresh masks button is pressed
    def on_mask_changed(self):
        if self.middle_mask_layer is None:
            return

        # Make a copy of the current mask layer
        current_mask = np.array(self.middle_mask_layer.data, copy=True)

        # Save new masks into the original stack and update UI
        new_ids = self.save_new_masks(current_mask)

        # Add checkboxes + delete buttons only for truly new masks
        for gid in new_ids:
            self.add_mask_row(gid)

        # Set selected_label to next free ID
        if self.current_mask_ids is not None:
            self.middle_mask_layer.selected_label = (
                max(self.current_mask_ids) + 1
            )
        else:
            self.middle_mask_layer.selected_label = 1

    # save masks that have been redrawn or added
    def save_new_masks(self, current_mask):

        # get all the mask ids in the current stack
        stack_mask = self.masks_stack[self.current_idx]
        stack_ids = np.unique(stack_mask)
        stack_ids = stack_ids[stack_ids != 0]

        current_ids = np.unique(current_mask)
        current_ids = current_ids[current_ids != 0]

        # find new mask ids given original id lists
        new_ids = [gid for gid in current_ids if gid not in stack_ids]

        # merge new masks into the saved stack
        for gid in new_ids:
            stack_mask[current_mask == gid] = gid

        # update the stack
        self.masks_stack[self.current_idx] = stack_mask

        # update the current IDs list (union of old + new)
        self.current_mask_ids = np.unique(stack_mask)
        self.current_mask_ids = self.current_mask_ids[
            self.current_mask_ids != 0
        ]

        return new_ids

    # add a single mask row (checkbox + delete button) to the scroll UI
    def add_mask_row(self, gid):

        row_layout = QHBoxLayout()

        # Checkbox
        cb = QCheckBox(f"Glom {gid}")
        cb.setChecked(True)  # start as visible
        cb.stateChanged.connect(
            lambda state, g=gid: self.toggle_mask(g, state)
        )
        row_layout.addWidget(cb)

        # Delete button
        btn = QPushButton("Delete")
        btn.clicked.connect(lambda _, g=gid: self.delete_mask(g))
        row_layout.addWidget(btn)

        # Add the row layout to the scroll layout
        self.checkboxLayout.addLayout(row_layout)

    # on nextBtn click
    def next_clicked(self):

        # check if there is anything loaded first
        if self.images_stack is None:
            return

        # save masks
        current_mask = np.array(self.middle_mask_layer.data, copy=True)
        self.save_new_masks(current_mask)

        # update current index
        self.current_idx = min(
            self.current_idx + 1, self.images_stack.shape[0] - 1
        )

        # update all panels/active layers
        self.update_panels()

    # on prevBtn click
    def prev_clicked(self):

        # check if there is anything loaded first
        if self.images_stack is None:
            return

        # save masks
        current_mask = np.array(self.middle_mask_layer.data, copy=True)
        self.save_new_masks(current_mask)

        # update current index
        self.current_idx = max(0, self.current_idx - 1)

        # update all panels/active layers
        self.update_panels()

    # on hideAllBtn click
    def hide_clicked(self):
        if self.middle_mask_layer is None:
            return

        # Create a new zero array of the same shape and dtype
        new_mask = np.zeros_like(
            np.array(self.middle_mask_layer.data, copy=True)
        )

        # Assign it back to the Labels layer to trigger redraw
        self.middle_mask_layer.data = new_mask

        # uncheck all checkboxes
        for i in range(self.checkboxLayout.count()):
            row_layout = self.checkboxLayout.itemAt(i)
            if row_layout is not None:
                for j in range(row_layout.count()):
                    w = row_layout.itemAt(j).widget()
                    if isinstance(w, QCheckBox):
                        w.setChecked(False)
