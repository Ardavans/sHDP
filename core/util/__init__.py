import os, sys
scriptpath = os.path.abspath(os.path.join(os.path.dirname(__file__))) #/core/util
sys.path.append(scriptpath)
__all__ = ['general','plot','stats','text']
import general, plot, stats, text
