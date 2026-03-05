# -*- mode: python ; coding: utf-8 -*-


import xgboost, os as _os
_xgb_dir = _os.path.dirname(xgboost.__file__)
_xgb_lib = _os.path.join(_xgb_dir, 'lib', 'libxgboost.dylib')
_xgb_ver = _os.path.join(_xgb_dir, 'VERSION')

a = Analysis(
    ['AusSuperPredictor.py'],
    pathex=[],
    binaries=[(_xgb_lib, 'xgboost/lib')],
    datas=[(_xgb_ver, 'xgboost')],
    hiddenimports=['xgboost', 'xgboost.core', 'xgboost.tracker'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AusSuperPredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AusSuperPredictor',
)
app = BUNDLE(
    coll,
    name='AusSuperPredictor.app',
    icon='resources/asx200predictor.icns',
    bundle_identifier=None,
)
