#!/usr/bin/env python

colormaps = [
    'Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'Dark2',
    'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn',
    'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu',
    'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy',
    'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2',
    'Set3', 'Spectral', 'YlGn', 'YlGnBu', 'YlOrBr',
    'YlOrRd', 'autumn', 'binary', 'black-white',
    'blue-red', 'bone', 'cool', 'copper', 'file',
    'flag', 'gist_earth', 'gist_gray', 'gist_heat',
    'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg',
    'gray', 'hot', 'hsv', 'jet', 'pink', 'prism', 'spectral',
    'spring', 'summer', 'winter'][::-1]

def traitcolor_to_mayavi(tcolor):
    return (tcolor[0]/255., tcolor[1]/255., tcolor[2]/255.)

def mayavi_to_traitcolor(mcolor):
    return (int(mcolor[0]*255), int(mcolor[1]*255), int(mcolor[2]*255), 255)