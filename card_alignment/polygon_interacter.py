import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
import cv2


class PolygonInteractor(object):
    """
    An polygon editor
    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly, img):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        self.img = img
        canvas = poly.figure.canvas
        self.poly = poly # rescale image

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)

        cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def get_poly_points(self):
        return np.asarray(self.poly.xy)

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state
        

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None
        try:
            cv2.destroyWindow("Zoom")
        except:
            pass

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

        self.display_zoom(x=x, y=y)
    
    def display_zoom(self, x, y):
        IM_HEIGHT, IM_WIDTH, _ = self.img.shape
        SIZE = 100
        RATIO_1 = 2
        ZOOM_HEIGHT = int(IM_HEIGHT/RATIO_1)
        RATIO_2 = ZOOM_HEIGHT/SIZE/2
        x, y = int(x), int(y)
        x1, y1, x2, y2 = x - 50, y - 50, x + 50, y + 50
        if x1 < 0:
            x1 = 0
            x2 = x1 + 100
            x = int(x * RATIO_1 * RATIO_2)
        elif x2 > IM_WIDTH:
            x2 = IM_WIDTH
            x1 = x2 - 100
            x = int((x - IM_WIDTH + 100 - 1) * RATIO_1 * RATIO_2)
        else:
            x = ZOOM_HEIGHT//RATIO_1 - 1
        if y1 < 0:
            y1 = 0
            y2 = y1 + 100
            y = int(y * RATIO_1 * RATIO_2)
        elif y2 > IM_HEIGHT:
            y2 = IM_HEIGHT
            y1 = y2 - 100
            y = int((y - IM_HEIGHT + 100 - 1) * RATIO_1 * RATIO_2)
        else:
            y = ZOOM_HEIGHT//RATIO_1 - 1
        zoom = self.img[y1:y2, x1:x2]
        zoom =  cv2.resize(zoom,(ZOOM_HEIGHT, ZOOM_HEIGHT))
        zoom = cv2.circle(zoom, (x, y), radius=10, color=(0, 0, 255), thickness=2)
        zoom = cv2.line(zoom, (x, max(y - 6, 0 )), (x, min(y + 6, ZOOM_HEIGHT)), color=(0, 0, 255), thickness=2)
        zoom = cv2.line(zoom, (max(x - 6, 0 ), y), (min(x + 6, ZOOM_HEIGHT), y), color=(0, 0, 255), thickness=2)
        cv2.imshow("Zoom", zoom)


