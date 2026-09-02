"""Small reusable Qt widgets/helpers shared across tabs."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QSizePolicy


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert an 8-bit grayscale or RGB numpy array into a QImage."""
    arr = np.ascontiguousarray(arr)
    height, width = arr.shape[:2]
    if arr.ndim == 2:
        qimage = QImage(arr.data, width, height, width, QImage.Format_Grayscale8)
    else:
        qimage = QImage(arr.data, width, height, width * 3, QImage.Format_RGB888)
    return qimage.copy()


class ZoomableImageView(QGraphicsView):
    """An image viewer with mouse-wheel zoom and click-drag panning.

    Shows one pixmap at a time, fit to the view by default. `set_image(...,
    reset_view=False)` lets a caller redraw the same image (e.g. after an
    adjustment) without losing the viewer's current zoom/pan.
    """

    zoom_changed = Signal(int)  # current zoom, as a percentage

    MIN_ZOOM = 0.05
    MAX_ZOOM = 32.0
    ZOOM_STEP = 1.15

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._has_image = False
        self._placeholder_text = ""

        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setBackgroundBrush(QColor("#202020"))
        self.setFrameShape(QGraphicsView.NoFrame)

    # ---------- content ----------
    def set_image(self, pixmap: QPixmap, reset_view: bool = True) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._has_image = True
        if reset_view:
            self.fit_to_window()
        else:
            self._emit_zoom_changed()
        self.viewport().update()

    def clear_image(self, text: str = "") -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())
        self._has_image = False
        self._placeholder_text = text
        self.resetTransform()
        self.viewport().update()

    # ---------- zoom ----------
    def fit_to_window(self) -> None:
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._emit_zoom_changed()

    def zoom_actual_size(self) -> None:
        self._set_absolute_zoom(1.0, anchor=QGraphicsView.AnchorViewCenter)

    def zoom_in(self) -> None:
        self._step_zoom(self.ZOOM_STEP, anchor=QGraphicsView.AnchorViewCenter)

    def zoom_out(self) -> None:
        self._step_zoom(1.0 / self.ZOOM_STEP, anchor=QGraphicsView.AnchorViewCenter)

    def current_zoom_percent(self) -> int:
        return round(self.transform().m11() * 100)

    def wheelEvent(self, event) -> None:
        if not self._has_image:
            return
        factor = self.ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / self.ZOOM_STEP
        self._step_zoom(factor, anchor=QGraphicsView.AnchorUnderMouse)

    def _step_zoom(self, factor: float, anchor) -> None:
        current = self.transform().m11()
        self._set_absolute_zoom(current * factor, anchor=anchor)

    def _set_absolute_zoom(self, target: float, anchor) -> None:
        if not self._has_image:
            return
        current = self.transform().m11()
        if current <= 0:
            return
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, target))
        self.setTransformationAnchor(anchor)
        self.scale(target / current, target / current)
        self._emit_zoom_changed()

    def _emit_zoom_changed(self) -> None:
        self.zoom_changed.emit(self.current_zoom_percent())

    # ---------- placeholder ----------
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._has_image or not self._placeholder_text:
            return
        painter = QPainter(self.viewport())
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(
            self.viewport().rect(), Qt.AlignCenter | Qt.TextWordWrap, self._placeholder_text
        )
        painter.end()
