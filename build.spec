# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：单文件、无控制台、内嵌 index.html 与离线模型。"""

block_cipher = None

a = Analysis(
    ['中草药识别.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('index.html', '.'),
        ('herb_model.tflite', '.'),
        ('herb_labels.txt', '.'),
    ],
    hiddenimports=[
        'ai_edge_litert',
        'ai_edge_litert.interpreter',
        'webview.platforms.edgechromium',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow', 'torch', 'matplotlib', 'scipy', 'pandas',
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='中草药识别',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
