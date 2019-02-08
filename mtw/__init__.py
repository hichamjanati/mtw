"""
Multitask Learning module for Python
====================================

"""
from .estimators import STL, Dirty, MTW, MLL, AdaSTL
from . import model_selection, ot
from .utils import (generate_dirac_images, get_std_blurr, blurrdata, get_std,
                    get_design_matrix, groundmetric2d, groundmetric)


__all__ = ['MTW', 'Dirty', 'STL', 'MLL', 'AdaSTL',
           "generate_dirac_images", "get_std_blurr", "blurrdata", "get_std",
           "get_design_matrix", "groundmetric", "groundmetric2d", "ot",
           "model_selection"]
