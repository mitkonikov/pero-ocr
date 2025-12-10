import numpy as np
from typing import List
from pero_ocr.core.layout import PageLayout
from pero_ocr.utils import config_get_list
from pero_ocr.layout_engines import layout_helpers as helpers

class XYCutRegionSorter:
    def __init__(self, config, config_path=None):
        self.config = config

        # optional categories to ignore/handle specially (same behaviour as SmartRegionSorter)
        try:
            self.categories = config_get_list(config, key='CATEGORIES', fallback=[])
        except Exception:
            self.categories = []

    def process_page(self, image, layout: PageLayout) -> PageLayout:
        # split page layout by categories (keeps region categories separate)
        page_layout, page_layout_ignore = helpers.split_page_layout_by_categories(layout, self.categories)

        # if there are too few regions, return merged layout (nothing to sort)
        if len(page_layout.regions) < 2:
            return helpers.merge_page_layouts(page_layout_ignore, page_layout)

        # Build bounding boxes array from regions: [x0, y0, x1, y1]
        boxes = []
        for reg in page_layout.regions:
            xs = reg.polygon[:, 0]
            ys = reg.polygon[:, 1]
            x0 = int(np.floor(xs.min()))
            y0 = int(np.floor(ys.min()))
            x1 = int(np.ceil(xs.max()))
            y1 = int(np.ceil(ys.max()))
            boxes.append([x0, y0, x1, y1])

        boxes = np.array(boxes, dtype=int)
        indices = np.arange(len(page_layout.regions), dtype=int)

        # recursive xy-cut will populate res with ordered indices
        res = []
        try:
            self.recursive_xy_cut(boxes, indices, res)
        except Exception as e:
            print(f'XY-Cut sorting failed with error: {e}')
            res = None

        # if recursive_xy_cut produced nothing, fallback
        if not res:
            return helpers.merge_page_layouts(page_layout_ignore, page_layout)

        # reorder regions according to res
        ordered_regions = [page_layout.regions[i] for i in res]
        page_layout.regions = ordered_regions

        # merge ignored categories back and return
        page_layout = helpers.merge_page_layouts(page_layout_ignore, page_layout)
        return page_layout
    
    @staticmethod
    def projection_by_bboxes(boxes: np.array, axis: int) -> np.ndarray:
        """
        Get the projection histogram through a set of bboxes and finally output it in per-pixel form

        Args:
            boxes: [N, 4]
            axis:   0-x coordinate is projected horizontally,
                    1-y coordinate is projected vertically

        Returns:
            1D projection histogram, where the length is the maximum value of the projection direction coordinate
            (we don't need the actual side length of the image, because we only want to find the spacing of the text boxes)
        """

        assert axis in [0, 1]
        length = np.max(boxes[:, axis::2])
        res = np.zeros(length, dtype=int)
        # TODO: how to remove for loop?
        for start, end in boxes[:, axis::2]:
            res[start:end] += 1
        return res

    # from: https://dothinking.github.io/2021-06-19-%E9%80%92%E5%BD%92%E6%8A%95%E5%BD%B1%E5%88%86%E5%89%B2%E7%AE%97%E6%B3%95/#:~:text=%E9%80%92%E5%BD%92%E6%8A%95%E5%BD%B1%E5%88%86%E5%89%B2%EF%BC%88Recursive%20XY,%EF%BC%8C%E5%8F%AF%E4%BB%A5%E5%88%92%E5%88%86%E6%AE%B5%E8%90%BD%E3%80%81%E8%A1%8C%E3%80%82
    @staticmethod
    def split_projection_profile(arr_values: np.array, min_value: float, min_gap: float):
        """Split projection profile:

        ```
                                ┌──┐
            arr_values           │  │       ┌─┐───
                ┌──┐             │  │       │ │ |
                │  │             │  │ ┌───┐ │ │min_value
                │  │<- min_gap ->│  │ │   │ │ │ |
            ────┴──┴─────────────┴──┴─┴───┴─┴─┴─┴───
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        ```

        Args:
            arr_values (np.array): 1-d array representing the projection profile.
            min_value (float): Ignore the profile if `arr_value` is less than `min_value`.
            min_gap (float): Ignore the gap if less than this value.

        Returns:
            tuple: Start indexes and end indexes of split groups.
        """
        # all indexes with projection height exceeding the threshold
        arr_index = np.where(arr_values > min_value)[0]
        if not len(arr_index):
            return

        # find zero intervals between adjacent projections
        # |  |                    ||
        # ||||<- zero-interval -> |||||
        arr_diff = arr_index[1:] - arr_index[0:-1]
        arr_diff_index = np.where(arr_diff > min_gap)[0]
        arr_zero_intvl_start = arr_index[arr_diff_index]
        arr_zero_intvl_end = arr_index[arr_diff_index + 1]

        # convert to index of projection range:
        # the start index of zero interval is the end index of projection
        arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
        arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
        arr_end += 1  # end index will be excluded as index slice

        return arr_start, arr_end

    @staticmethod
    def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int]):
        """
        Args:
            boxes: (N, 4)
            indices: The recursive process always represents the index of the box in the original data
            res: Save the output results

        """
        # Projection onto the Y-axis
        assert len(boxes) == len(indices)

        _indices = boxes[:, 1].argsort()
        y_sorted_boxes = boxes[_indices]
        y_sorted_indices = indices[_indices]

        y_projection = XYCutRegionSorter.projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
        pos_y = XYCutRegionSorter.split_projection_profile(y_projection, 0, 1)
        if not pos_y:
            return

        arr_y0, arr_y1 = pos_y
        for r0, r1 in zip(arr_y0, arr_y1):
            # [r0, r1] means that the area with bbox will be split
            # vertically according to the horizontal segmentation.
            _indices = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)

            y_sorted_boxes_chunk = y_sorted_boxes[_indices]
            y_sorted_indices_chunk = y_sorted_indices[_indices]

            _indices = y_sorted_boxes_chunk[:, 0].argsort()
            x_sorted_boxes_chunk = y_sorted_boxes_chunk[_indices]
            x_sorted_indices_chunk = y_sorted_indices_chunk[_indices]

            # Projection in the X direction
            x_projection = XYCutRegionSorter.projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
            pos_x = XYCutRegionSorter.split_projection_profile(x_projection, 0, 1)
            if not pos_x:
                continue

            arr_x0, arr_x1 = pos_x
            if len(arr_x0) == 1:
                # The x direction cannot be split
                # Take the result from sorted by y direction,
                # since reading order is top to bottom, left to right
                res.extend(y_sorted_indices_chunk)
                continue

            # Can be separated in the x direction, continue to call recursively
            for c0, c1 in zip(arr_x0, arr_x1):
                _indices = (c0 <= x_sorted_boxes_chunk[:, 0]) & (
                    x_sorted_boxes_chunk[:, 0] < c1
                )
                XYCutRegionSorter.recursive_xy_cut(
                    x_sorted_boxes_chunk[_indices], x_sorted_indices_chunk[_indices], res
                )
