"""Exact valid-pixel queries over a compact block-count spatial index."""
import heapq
import math

import numpy as np


class ValidMaskIndex:
    """Keep counts per 64x64 block, inspecting pixels only at query boundaries.

    The prefix sums are over blocks, not pixels (about 1/4096 of mask size).
    Nearest-point search descends occupied blocks in distance order and keeps
    the original row/column tie break. The underlying mask may be a memmap.
    """
    block_size = 64

    def __init__(self, mask):
        self.mask = mask
        self.height, self.width = mask.shape
        side = self.block_size
        self.counts = np.zeros(((self.height + side - 1)//side,
                                (self.width + side - 1)//side), dtype=np.int64)
        self.prefix = None

    def add_window(self, x, y, valid):
        """Accumulate aligned windows during the existing raster validity scan."""
        side = self.block_size
        if x % side or y % side:
            raise ValueError('Mask-index windows must align to index blocks.')
        h, w = valid.shape
        counts = np.add.reduceat(valid, np.arange(0, h, side), axis=0, dtype=np.int64)
        counts = np.add.reduceat(counts, np.arange(0, w, side), axis=1, dtype=np.int64)
        self.counts[y//side:y//side+len(counts), x//side:x//side+counts.shape[1]] = counts

    def finish(self):
        self.prefix = np.pad(self.counts.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        return self

    @classmethod
    def build(cls, mask):
        index = cls(mask)
        for y in range(0, mask.shape[0], 512):
            for x in range(0, mask.shape[1], 1024):
                index.add_window(x, y, mask[y:y+512, x:x+1024])
        return index.finish()

    def _block_count(self, x0, y0, x1, y1):
        p = self.prefix
        return int(p[y1, x1] - p[y0, x1] - p[y1, x0] + p[y0, x0])

    def _pixels(self, bounds):
        """Count boundary strips in bounded chunks, never a giant temporary."""
        x0, y0, x1, y1 = bounds
        count = 0
        for y in range(y0, y1, 256):
            for x in range(x0, x1, 4096):
                count += int(np.count_nonzero(self.mask[y:min(y+256, y1), x:min(x+4096, x1)]))
        return count

    def count(self, bounds):
        x0, y0, x1, y1 = bounds
        if x0 >= x1 or y0 >= y1:
            return 0
        side = self.block_size
        bx0, by0 = (x0+side-1)//side, (y0+side-1)//side
        bx1, by1 = x1//side, y1//side
        if bx0 >= bx1 or by0 >= by1:
            return self._pixels(bounds)
        left, top, right, bottom = bx0*side, by0*side, bx1*side, by1*side
        return (self._block_count(bx0, by0, bx1, by1)
                + self._pixels((x0, y0, x1, top))
                + self._pixels((x0, bottom, x1, y1))
                + self._pixels((x0, top, left, bottom))
                + self._pixels((right, top, x1, bottom)))

    def occupied(self, bounds):
        count = self.count(bounds)
        if not count:
            return 0, bounds
        x0, y0, x1, y1 = bounds
        # Tighten each boundary by exact count queries, without rescanning the
        # interior. Full rectangles need no boundary search at all.
        if count == (x1-x0)*(y1-y0):
            return count, bounds
        lo, hi = x0, x1-1
        while lo < hi:
            mid = (lo+hi)//2
            if self.count((x0, y0, mid+1, y1)): hi = mid
            else: lo = mid+1
        left = lo
        lo, hi = left, x1-1
        while lo < hi:
            mid = (lo+hi+1)//2
            if self.count((mid, y0, x1, y1)): lo = mid
            else: hi = mid-1
        right = lo+1
        lo, hi = y0, y1-1
        while lo < hi:
            mid = (lo+hi)//2
            if self.count((left, y0, right, mid+1)): hi = mid
            else: lo = mid+1
        top = lo
        lo, hi = top, y1-1
        while lo < hi:
            mid = (lo+hi+1)//2
            if self.count((left, mid, right, y1)): lo = mid
            else: hi = mid-1
        return count, (left, top, right, lo+1)

    def nearest(self, bounds, pixel_size):
        x0, y0, x1, y1 = bounds
        cx, cy = (x0+x1-1)/2, (y0+y1-1)/2
        sx, sy = pixel_size
        side = self.block_size
        best, point = math.inf, None
        heap = []

        def push(bx0, by0, bx1, by1):
            if not self._block_count(bx0, by0, bx1, by1):
                return
            left, top = max(x0, bx0*side), max(y0, by0*side)
            right, bottom = min(x1, bx1*side)-1, min(y1, by1*side)-1
            if left > right or top > bottom:
                return
            dx, dy = max(left-cx, 0, cx-right)*sx, max(top-cy, 0, cy-bottom)*sy
            lower = dx*dx + dy*dy
            if lower <= best:
                heapq.heappush(heap, (lower, by0, bx0, by1, bx1))

        push(x0//side, y0//side, (x1+side-1)//side, (y1+side-1)//side)
        while heap:
            lower, by0, bx0, by1, bx1 = heapq.heappop(heap)
            if lower > best:
                break
            if bx1-bx0 == 1 and by1-by0 == 1:
                left, top = max(x0, bx0*side), max(y0, by0*side)
                right, bottom = min(x1, bx1*side), min(y1, by1*side)
                valid = self.mask[top:bottom, left:right]
                if not valid.any():
                    continue
                distances = (((np.arange(left, right)-cx)*sx)**2)[None, :] + (((np.arange(top, bottom)-cy)*sy)**2)[:, None]
                distances[~valid] = np.inf
                index = int(np.argmin(distances))
                row, col = np.unravel_index(index, distances.shape)
                candidate = (top+int(row), left+int(col))
                distance = float(distances.flat[index])
                if distance < best or (distance == best and (point is None or candidate < point)):
                    best, point = distance, candidate
            elif bx1-bx0 >= by1-by0 and bx1-bx0 > 1:
                middle = (bx0+bx1)//2
                push(bx0, by0, middle, by1)
                push(middle, by0, bx1, by1)
            else:
                middle = (by0+by1)//2
                push(bx0, by0, bx1, middle)
                push(bx0, middle, bx1, by1)
        return point
