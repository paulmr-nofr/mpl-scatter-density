from math import log10

import numpy as np

from fast_histogram import histogram2d


class FixedDataDensityHelper:

    compute_when_pressed = True

    def __init__(self, ax, x, y, c=None, downres_factor=4, linthresh=None, linscale=None, base=10):
        self._ax = ax
        self._c = None
        self._downres = False
        self._linthresh = None
        self._linscale = None
        self._base = 10

        if downres_factor < 1 or downres_factor % 1 != 0:
            raise ValueError('downres_factor should be a strictly positive integer value')

        self._downres_factor = downres_factor
        self.set_xy(x, y)
        self.set_c(c)
        self.set_linthresh(linthresh)
        self.set_linscale(linscale)
        self.set_base(base)

    def downres(self):
        self._downres = True

    def upres(self):
        self._downres = False

    def set_xy(self, x, y):
        self._x = x
        self._y = y
        self._x_log = None
        self._x_symlog = None
        self._y_log = None
        self._y_symlog = None
        self._x_log_sub = None
        self._x_symlog_sub = None
        self._y_log_sub = None
        self._y_symlog_sub = None
        step = self._downres_factor ** 2
        self._x_sub = self._x[::step]
        self._y_sub = self._y[::step]

    def set_c(self, c):
        self._c = c
        step = self._downres_factor ** 2
        if self._c is None:
            self._c_sub = None
        else:
            self._c_sub = self._c[::step]
    
    def set_linthresh(self, linthresh):
        self._linthresh = linthresh
    
    def set_linscale(self, linscale):
        self._linscale = linscale
    
    def set_base(self, base):
        self._base = base

    def _update_x_log(self):
        step = self._downres_factor ** 2
        with np.errstate(invalid='ignore'):
            self._x_log = np.log10(self._x)
        self._x_log_sub = self._x_log[::step]

    def _update_y_log(self):
        step = self._downres_factor ** 2
        with np.errstate(invalid='ignore'):
            self._y_log = np.log10(self._y)
        self._y_log_sub = self._y_log[::step]

    def _update_x_symlog(self):
        step = self._downres_factor ** 2
        abs_a = np.abs(self._x)
        with np.errstate(divide="ignore", invalid="ignore"):
            linscale_adj = (self._linscale / (1.0 - self._base ** -1))
            self._x_symlog = np.sign(self._x) * self._linthresh * (linscale_adj + np.log(abs_a / self._linthresh) / np.log(self._base))
            inside = abs_a <= self._linthresh
            self._x_symlog[inside] = self._x[inside] * linscale_adj
        self._x_symlog_sub = self._x_symlog[::step]

    def _update_y_symlog(self):
        step = self._downres_factor ** 2
        abs_a = np.abs(self._y)
        with np.errstate(divide="ignore", invalid="ignore"):
            linscale_adj = (self._linscale / (1.0 - self._base ** -1))
            self._y_symlog = np.sign(self._y) * self._linthresh * (linscale_adj + np.log(abs_a / self._linthresh) / np.log(self._base))
            inside = abs_a <= self._linthresh
            self._y_symlog[inside] = self._y[inside] * linscale_adj
        self._y_symlog_sub = self._y_symlog[::step]
    

    def __call__(self, bins=None, range=None):

        ny, nx = bins
        (ymin, ymax), (xmin, xmax) = range

        xscale = self._ax.get_xscale()
        yscale = self._ax.get_yscale()

        if xscale == 'log':
            xmin, xmax = log10(xmin), log10(xmax)
            if self._x_log is None:
                # We do this here insead of in set_xy to save time since in
                # set_xy we don't know yet if the axes will be log or not.
                self._update_x_log()
            if self._downres:
                x = self._x_log_sub
            else:
                x = self._x_log
        
        elif xscale == 'symlog':
            xmax =  np.sign(xmax) * self._linthresh * (linscale_adj + np.log(np.abs(xmax) / self._linthresh) / np.log(self._base))
            xmin =  np.sign(xmin) * self._linthresh * (linscale_adj + np.log(np.abs(xmin) / self._linthresh) / np.log(self._base))
            if self._x_symlog is None:
                # We do this here insead of in set_xy to save time since in
                # set_xy we don't know yet if the axes will be log or not.
                self._update_x_symlog()
            if self._downres:
                x = self._x_symlog_sub
            else:
                x = self._x_symlog
        
        elif xscale == 'linear':
            if self._downres:
                x = self._x_sub
            else:
                x = self._x
        else:  # pragma: nocover
            raise ValueError('Unexpected xscale: {0}'.format(xscale))

        if yscale == 'log':
            ymin, ymax = log10(ymin), log10(ymax)
            if self._y_log is None:
                # We do this here insead of in set_xy to save time since in
                # set_xy we don't know yet if the axes will be log or not.
                self._update_y_log()
            if self._downres:
                y = self._y_log_sub
            else:
                y = self._y_log

        elif yscale == 'symlog':
            linscale_adj = (self._linscale / (1.0 - self._base ** -1))
            ymax =  np.sign(ymax) * self._linthresh * (linscale_adj + np.log(np.abs(ymax) / self._linthresh) / np.log(self._base))
            ymin =  np.sign(ymin) * self._linthresh * (linscale_adj + np.log(np.abs(ymin) / self._linthresh) / np.log(self._base))
            if self._y_symlog is None:
                # We do this here insead of in set_xy to save time since in
                # set_xy we don't know yet if the axes will be log or not.
                self._update_y_symlog()
            if self._downres:
                y = self._y_symlog_sub
            else:
                y = self._y_symlog

        elif yscale == 'linear':
            if self._downres:
                y = self._y_sub
            else:
                y = self._y
        else:  # pragma: nocover
            raise ValueError('Unexpected xscale: {0}'.format(yscale))

        if self._downres:
            nx_sub = nx // self._downres_factor
            ny_sub = ny // self._downres_factor
            bins = (ny_sub, nx_sub)
            weights = self._c_sub
        else:
            bins = (ny, nx)
            weights = self._c

        if weights is None:
            array = histogram2d(y, x, bins=bins, range=((ymin, ymax), (xmin, xmax)))
        else:
            array = histogram2d(y, x, bins=bins, weights=weights,
                                range=((ymin, ymax), (xmin, xmax)))
            count = histogram2d(y, x, bins=bins,
                                range=((ymin, ymax), (xmin, xmax)))

            with np.errstate(invalid='ignore'):
                array /= count

        return array
