from typing import TYPE_CHECKING

import dask.array as da
import imageio.v3 as iio
import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from pathlib import Path

if TYPE_CHECKING:
    import napari


class SegmentationEditor(QWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.current_idx = 0
        self.total_slices = 0

        self.image_stacks: list[da.Array] = None
        # Masks stack remains a List[np.ndarray] (Mutable)
        self.masks_stack: list[np.ndarray] = None

        # Layer references
        self.middle_mask_layer = None
        self.prev_image_layer = None
        self.middle_image_layer = None
        self.next_image_layer = None

        self.visible_ids = None

        self.current_mask_ids = []
        self.mask_filenames: list[str] = []

        self.setAcceptDrops(True)

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

        self.showAllBtn = QPushButton("Show All")
        self.showAllBtn.clicked.connect(self.show_clicked)

        self.exportStackBtn = QPushButton("Export Masks as Stack")
        self.exportStackBtn.clicked.connect(self.export_masks)

        self.exportIndivBtn = QPushButton("Export Masks as Images")
        self.exportIndivBtn.clicked.connect(self.export_individual_masks)

        """ Layouts """
        mainLayout = QVBoxLayout()
        scrollLayout = QHBoxLayout()
        scrollLayout.addWidget(self.prevBtn)
        scrollLayout.addWidget(self.nextBtn)

        refLayout = QHBoxLayout()
        refLayout.addWidget(self.refreshBtn)

        togLayout = QHBoxLayout()
        togLayout.addWidget(self.hideAllBtn)
        togLayout.addWidget(self.showAllBtn)

        exportLayout = QVBoxLayout()
        exportLayout.addWidget(self.exportStackBtn)
        exportLayout.addWidget(self.exportIndivBtn)

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
        mainLayout.addLayout(exportLayout)

        self.setLayout(mainLayout)

    # --- DATA LOADING --

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")

        if not folder: return

        from ._reader import reader_function

        # reader_function returns a list of tuples: [(img1_da, ...), (img2_da, ...), ..., (mask_da, ...)]
        layer_data = reader_function(folder)

        mask_tuple = layer_data.pop() 
        masks_stack = mask_tuple[0]
        self.mask_filenames = mask_tuple[1].get("filenames", [])

        # Remaining items are image stacks
        image_stacks = [data[0] for data in layer_data]

        '''
        # Extract only the Dask arrays (index [0] of each tuple)
        all_dask_stacks = [data[0] for data in layer_data]

        # pop the last item (which is always the mask stack)
        masks_stack = all_dask_stacks.pop()

        # The remaining items are the image channel stacks (now a list of Dask arrays)
        image_stacks = all_dask_stacks
        '''

        # Pass the list of image stacks and the single mask stack to load_data
        self.load_data(image_stacks, masks_stack)

    def load_data(self, image_stacks: list[da.Array], masks_stack: da.Array):

        self.image_stacks = image_stacks  # List of Dask arrays
        self.current_idx = 0

        if self.image_stacks:
            self.total_slices = self.image_stacks[0].shape[0]
        else:
            self.total_slices = 0

        self.masks_stack = [
            masks_stack[i].compute() for i in range(masks_stack.shape[0])
        ]

        self.update_panels()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        folder = event.mimeData().urls()[0].toLocalFile()
        from ._reader import reader_function

        layer_data = reader_function(folder)
        all_dask_stacks = [data[0] for data in layer_data]

        masks_stack = all_dask_stacks.pop()
        image_stacks = all_dask_stacks

        self.load_data(image_stacks, masks_stack)

    # image stacking helper if more than one image folder is uploaded
    def _get_stacked_slice(self, index: int) -> np.ndarray:
        """Computes and stacks the slice at 'index' from all image channels into a (C, H, W) array."""
        if not self.image_stacks:
            # Fallback for empty image stack
            return np.zeros(self.masks_stack[0].shape, dtype=np.uint8)

        slices = [stack[index].compute() for stack in self.image_stacks]

        # Stack slices along a new axis (axis=0) for napari multi-channel view
        stacked_data = np.stack(slices, axis=0)
        return stacked_data

    # --- LAYER INITIALIZATION (RUNS ONCE) ---

    def _initialize_layers(self):
        """Creates all napari layers ONCE when data is first loaded."""

        if not self.image_stacks:
            QMessageBox.critical(self, "Init Error", "No image stacks loaded.")
            return

        single_slice_shape = self.masks_stack[0].shape
        C = len(self.image_stacks)
        zero_stack_shape = (C,) + single_slice_shape
        zero_dtype = self.image_stacks[0].dtype
        max_idx = self.total_slices - 1

        # 1. Previous Layer (start as invisible placeholder)
        prev_data = np.zeros(zero_stack_shape, dtype=zero_dtype)
        self.prev_image_layer = self.viewer.add_image(
            prev_data, name="Previous", visible=False
        )

        # 3. Next Layer
        if max_idx >= 1:
            next_data = self._get_stacked_slice(self.current_idx + 1)
            next_visible = True
            next_name = f"Pos{self.current_idx + 1} - Next"
        else:
            next_data = np.zeros(zero_stack_shape, dtype=zero_dtype)
            next_visible = False
            next_name = "Next (Placeholder)"

        self.next_image_layer = self.viewer.add_image(
            next_data, name=next_name, visible=next_visible
        )

        # 2. Current Layer (load the first frame)
        current_data = self._get_stacked_slice(self.current_idx)
        self.middle_image_layer = self.viewer.add_image(
            current_data, name=f"Pos{self.current_idx} - Current"
        )

        # 4. Current Mask Layer
        mask_data = self.masks_stack[self.current_idx]
        self.middle_mask_layer = self.viewer.add_labels(
            mask_data.copy(), name=f"Pos{self.current_idx} masks"
        )

        # Move mask layer to the top and select it
        self.viewer.layers.move(
            self.viewer.layers.index(self.middle_mask_layer),
            len(self.viewer.layers) - 1,
        )
        self.viewer.layers.selection.active = self.middle_mask_layer

        self._update_ui_state()

    # --- LAYER UPDATE (RUNS ON EVERY CLICK) ---

    def _update_layer_data(self):
        """Updates the data of existing layers (very fast)."""

        idx = self.current_idx
        max_idx = self.total_slices - 1

        # 1. Compute and assign data for Current Layer first (uses stacking helper)
        current_data = self._get_stacked_slice(idx)
        self.middle_image_layer.data = current_data
        self.middle_image_layer.name = f"Pos{idx} - Current"

        # Get settings from the layer the user is currently interacting with
        current_clim = self.middle_image_layer.contrast_limits
        current_cmap = self.middle_image_layer.colormap.name

        # 2. Previous Layer Update
        prev_idx = idx - 1
        if prev_idx >= 0:
            new_data = self._get_stacked_slice(prev_idx)
            self.prev_image_layer.data = new_data
            self.prev_image_layer.name = f"Pos{prev_idx} - Previous"
            self.prev_image_layer.visible = True

            # Apply CURRENT settings to PREVIOUS layer
            self.prev_image_layer.contrast_limits = current_clim
            self.prev_image_layer.colormap = current_cmap
        else:
            self.prev_image_layer.visible = False

        # 3. Next Layer Update
        next_idx = idx + 1
        if next_idx <= max_idx:
            new_data = self._get_stacked_slice(next_idx)
            self.next_image_layer.data = new_data
            self.next_image_layer.name = f"Pos{next_idx} - Next"
            self.next_image_layer.visible = True

            # Apply CURRENT settings to NEXT layer
            self.next_image_layer.contrast_limits = current_clim
            self.next_image_layer.colormap = current_cmap
        else:
            self.next_image_layer.visible = False

        # 4. Mask Layer Update
        self.middle_mask_layer.data = self.masks_stack[idx].copy()
        self.middle_mask_layer.name = f"Pos{idx} masks"

        self._update_ui_state()

    def _update_ui_state(self):
        """Helper to sync IDs, checkbox list, and next label ID."""

        self.current_mask_ids = np.unique(self.middle_mask_layer.data)
        self.current_mask_ids = self.current_mask_ids[
            self.current_mask_ids != 0
        ]

        self.update_mask_list()

        if (
            self.current_mask_ids is not None
            and self.current_mask_ids.size > 0
        ):
            next_id = max(self.current_mask_ids) + 1
        else:
            next_id = 1
        self.middle_mask_layer.selected_label = next_id

    # --- MAIN ENTRY POINT ---

    def update_panels(self):
        """Replaces the old, slow update_panels logic."""

        # CHANGED check for image_stacks
        if self.image_stacks is None or self.masks_stack is None:
            return

        if self.middle_image_layer is None:
            self._initialize_layers()
        else:
            self._update_layer_data()

    # --- NAVIGATION CLICKS ---

    def next_clicked(self):
        # check for image_stacks
        if self.image_stacks is None:
            return

        # Save current state before moving
        if self.middle_mask_layer is not None:
            current_mask = np.array(self.middle_mask_layer.data, copy=True)
            self.save_new_masks(current_mask)

        # use total_slices
        self.current_idx = min(self.current_idx + 1, self.total_slices - 1)

        self.update_panels()
        self.visible_ids = None

    def prev_clicked(self):
        # check for image_stacks
        if self.image_stacks is None:
            return

        # Save current state before moving
        if self.middle_mask_layer is not None:
            current_mask = np.array(self.middle_mask_layer.data, copy=True)
            self.save_new_masks(current_mask)

        self.current_idx = max(0, self.current_idx - 1)

        self.update_panels()
        self.visible_ids = None

    # --- MASK MUTABILITY & UI METHODS ---

    def save_new_masks(self, current_visible_mask):
        """
        Merges new edits from the visible layer with the old persistent mask data,
        saves the complete result, and tracks new IDs.
        """

        stack_mask_before = self.masks_stack[
            self.current_idx
        ]  # Current persistent mask

        stack_ids_before = np.unique(stack_mask_before)
        stack_ids_before = stack_ids_before[stack_ids_before != 0]

        # Get all IDs visible in the napari layer (newly painted or edited visible masks)
        ids_in_visible_layer = np.unique(current_visible_mask)
        ids_in_visible_layer = ids_in_visible_layer[ids_in_visible_layer != 0]

        # Start the new persistent mask with the old persistent mask (includes hidden masks)
        new_complete_mask = stack_mask_before.copy()

        # Merge new/edited regions from the visible layer into the complete mask.
        # This loop ensures that any new painting or editing overwrites the persistent data.
        for gid in ids_in_visible_layer:
            # Find the pixels in the visible layer belonging to this ID
            visible_region = current_visible_mask == gid

            # The new persistent mask retains the visible regions
            new_complete_mask[visible_region] = gid

        # Find all IDs that were in the stack, but are no longer tracked in the UI list
        ids_to_delete = [
            gid for gid in stack_ids_before if gid not in self.current_mask_ids
        ]
        for gid in ids_to_delete:
            new_complete_mask[new_complete_mask == gid] = 0

        # Replace the entire slice in the mutable list with the new, complete mask
        self.masks_stack[self.current_idx] = new_complete_mask

        # Determine new IDs for UI row addition
        stack_ids_after = np.unique(new_complete_mask)
        stack_ids_after = stack_ids_after[stack_ids_after != 0]
        new_ids = [
            gid for gid in stack_ids_after if gid not in stack_ids_before
        ]

        return new_ids

    def toggle_mask(self, gid, state):
        if self.middle_mask_layer is None:
            return

        current_visible_mask = np.array(self.middle_mask_layer.data, copy=True)
        # Use the persistent stack as the source of truth for restoration
        original_slice_data = self.masks_stack[self.current_idx]

        if state:
            # SHOW: Restore the pixels matching 'gid' from the original slice data
            current_visible_mask[original_slice_data == gid] = gid
        else:
            # HIDE: Set the pixels matching 'gid' in the current visible data to 0
            current_visible_mask[current_visible_mask == gid] = 0

        self.middle_mask_layer.data = current_visible_mask

    def delete_mask(self, gid):
        if self.middle_mask_layer is None:
            return

        # 1. Update the visible layer
        mask_data = np.array(self.middle_mask_layer.data, copy=True)
        mask_data[mask_data == gid] = 0
        self.middle_mask_layer.data = mask_data

        # 2. Update the saved mutable stack
        self.masks_stack[self.current_idx][
            self.masks_stack[self.current_idx] == gid
        ] = 0

        self.current_mask_ids = [g for g in self.current_mask_ids if g != gid]

        # Remove the corresponding row from the UI (unchanged logic)
        for i in range(self.checkboxLayout.count()):
            layout_item = self.checkboxLayout.itemAt(i)
            if layout_item is not None:
                for j in range(layout_item.count()):
                    w = layout_item.itemAt(j).widget()
                    if isinstance(w, QCheckBox) and w.text() == f"Glom {gid}":
                        while layout_item.count():
                            child = layout_item.takeAt(0)
                            if child.widget():
                                child.widget().deleteLater()
                        self.checkboxLayout.removeItem(layout_item)
                        break

    def on_mask_changed(self):
        if self.middle_mask_layer is None:
            return

        # 1. Capture the list of IDs that are currently visible BEFORE saving/refreshing.
        current_visible_ids = np.unique(self.middle_mask_layer.data)
        current_visible_ids = current_visible_ids[current_visible_ids != 0]

        # Store these so update_mask_list() can use them
        self.visible_ids = set(current_visible_ids.tolist())

        # 2. Trigger the SAVE and ID detection using the currently visible layer data.
        current_mask = np.array(self.middle_mask_layer.data, copy=True)
        new_ids = self.save_new_masks(current_mask)

        # 3. Add UI rows for any brand new IDs
        for gid in new_ids:
            self.add_mask_row(gid)

        # 4. Restore the middle mask layer to the full persistent state
        complete_mask = self.masks_stack[self.current_idx].copy()
        self.middle_mask_layer.data = complete_mask

        # 5. Update the UI state (this will REBUILD the checkbox list)
        self._update_ui_state()

        # 6. Re-hide masks that were previously hidden
        ids_to_hide = [
            gid
            for gid in np.unique(complete_mask)
            if gid != 0 and gid not in self.visible_ids
        ]

        if ids_to_hide:
            restored_visible_mask = complete_mask.copy()
            for gid in ids_to_hide:
                restored_visible_mask[restored_visible_mask == gid] = 0
            self.middle_mask_layer.data = restored_visible_mask

    def add_mask_row(self, gid):
        row_layout = QHBoxLayout()

        cb = QCheckBox(f"Glom {gid}")
        cb.setChecked(True)
        cb.stateChanged.connect(
            lambda state, g=gid: self.toggle_mask(g, state)
        )
        row_layout.addWidget(cb)

        btn = QPushButton("Delete")
        btn.clicked.connect(lambda _, g=gid: self.delete_mask(g))
        row_layout.addWidget(btn)

        self.checkboxLayout.addLayout(row_layout)

    def update_mask_list(self):
        # Clear old rows
        for i in reversed(range(self.checkboxLayout.count())):
            layout_item = self.checkboxLayout.itemAt(i)
            if layout_item is not None:
                for j in reversed(range(layout_item.count())):
                    w = layout_item.itemAt(j).widget()
                    if w:
                        w.setParent(None)
                self.checkboxLayout.removeItem(layout_item)

        # Extract IDs from the current slice
        mask = self.middle_mask_layer.data
        mask_ids = np.unique(mask)
        mask_ids = mask_ids[mask_ids != 0]
        self.current_mask_ids = mask_ids

        # Build checkbox rows
        for gid in mask_ids:
            row_layout = QHBoxLayout()

            cb = QCheckBox(f"Glom {gid}")

            # Use stored visible_ids (preserved across refresh)
            if self.visible_ids is not None:
                cb.setChecked(gid in self.visible_ids)
            else:
                cb.setChecked(True)  # default

            cb.stateChanged.connect(
                lambda state, g=gid: self.toggle_mask(g, state)
            )
            row_layout.addWidget(cb)

            btn = QPushButton("Delete")
            btn.clicked.connect(lambda _, g=gid: self.delete_mask(g))
            row_layout.addWidget(btn)

            self.checkboxLayout.addLayout(row_layout)

    def hide_clicked(self):
        if self.middle_mask_layer is None:
            return

        new_mask = np.zeros_like(
            np.array(self.middle_mask_layer.data, copy=True)
        )
        self.middle_mask_layer.data = new_mask

        for i in range(self.checkboxLayout.count()):
            row_layout = self.checkboxLayout.itemAt(i)
            if row_layout is not None:
                for j in range(row_layout.count()):
                    w = row_layout.itemAt(j).widget()
                    if isinstance(w, QCheckBox):
                        w.setChecked(False)

    def show_clicked(self):
        if self.middle_mask_layer is None:
            return

        original_slice_data = self.masks_stack[self.current_idx]
        self.middle_mask_layer.data = original_slice_data.copy()

        for i in range(self.checkboxLayout.count()):
            row_layout = self.checkboxLayout.itemAt(i)
            if row_layout is not None:
                for j in range(row_layout.count()):
                    w = row_layout.itemAt(j).widget()
                    if isinstance(w, QCheckBox):
                        w.setChecked(True)

    # --- EXPORT METHODS ---

    def export_masks(self):
        """Exports the list of edited NumPy arrays back to a stack and saves as TIFF (Stack Export)."""

        if self.middle_mask_layer is not None:
            current_mask = np.array(self.middle_mask_layer.data, copy=True)
            self.save_new_masks(current_mask)

        if not self.masks_stack:
            QMessageBox.warning(
                self,
                "Export Error",
                "No mask data loaded or edited to export.",
            )
            return

        file_filter = "TIFF Stack (*.tif);;PNG Image (*.png)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Edited Masks (Stack)", "", file_filter
        )
        if not path:
            return

        try:
            final_numpy_stack = np.stack(self.masks_stack, axis=0)

            if ".png" in path.lower() and final_numpy_stack.shape[0] > 1:
                path = path.replace(".png", ".tif")
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Multi-slice stack saved as a multi-page TIFF file instead of PNG.",
                )

            iio.imwrite(path, final_numpy_stack.astype(np.uint32))

            QMessageBox.information(
                self,
                "Export Successful",
                f"Masks saved as a single stack to:\n{path}",
            )

        except (OSError, ValueError) as e:
            QMessageBox.critical(
                self, "Export Failed", f"An error occurred during save:\n{e}"
            )

    def export_individual_masks(self):
        """Saves each slice in self.masks_stack as a separate file."""
        
        # sync current edits first
        if self.middle_mask_layer is not None:
            current_mask = np.array(self.middle_mask_layer.data, copy=True)
            self.save_new_masks(current_mask)

        if not self.masks_stack or not self.mask_filenames:
            QMessageBox.warning(self, "Export Error", "No mask data or filenames found.")
            return
        
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select or Create Output Folder")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setLabelText(QFileDialog.Accept, "Save") 

        if dialog.exec_():
            # Get the path from THIS dialog
            selected_dirs = dialog.selectedFiles()
            if not selected_dirs:
                return
            output_path = Path(selected_dirs[0])
        else:
            return # User canceled

        try:
            for i, mask_slice in enumerate(self.masks_stack):
                filename = self.mask_filenames[i]
                save_path = output_path / filename
                
                # Use imageio to write the individual numpy array
                iio.imwrite(save_path, mask_slice.astype(np.uint32))

            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Saved {len(self.masks_stack)} masks to:\n{output_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred:\n{e}")