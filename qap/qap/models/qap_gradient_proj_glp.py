# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: models
# @file: /qap_gradient_proj.py
# @created: Sunday, 27th September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Sunday, 27th September 2020 4:46:00 pm
# @description:
#  The module for Goldstein-Levitin-Poljak
#    projected gradient method.
#  The key is to compute projections

from .qap_utils import *

logger = logging.getLogger('qap.run.gradient_projection_glp')
