import warnings

# Tesnsorflow has all sorts of internal dependencies that are on the deprecation
# path. We're not going to fix them without upgrading tensorflow, so let's
# ignore them to reduce the noise.
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
