from .DANet import build_DANet
_GZSL_META_ARCHITECTURES = {
    "Model": build_build_DANet,
}
def build_gzsl_pipeline(cfg):
    meta_arch = _GZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
