"""
config.py — All hyperparameters and settings in one place.
Edit this file to tune the model.
"""

CONFIG = {
    # ── Data ─────────────────────────────────────────────────
    'period':           '7y',
    'seq_len':          60,
    'predict_steps':    1,
    'train_ratio':      0.70,
    'val_ratio':        0.15,
    # test_ratio = 1 - train - val = 0.15

    # ── Model ────────────────────────────────────────────────
    'inter_neurons':    96,
    'command_neurons':  48,
    'motor_neurons':    24,
    'num_layers':       2,
    'dropout':          0.4,
    'ode_unfolds':      3,

    # ── Training ─────────────────────────────────────────────
    'batch_size':       1024,
    'epochs':           150,
    'lr':               5e-4,
    'weight_decay':     1e-4,
    'grad_clip':        1.0,
    'patience':         25,
    'ensemble_seeds':   [42, 123, 777],
    'confidence_thresh': 0.60,

    # ── Features ─────────────────────────────────────────────
    'n_features_mi':    50,       # top-N by mutual information
    'corr_threshold':   0.95,     # drop features correlated above this

    # ── Paths ────────────────────────────────────────────────
    'save_dir':         'saved_models',
    'results_dir':      'results',
    'plots_dir':        'plots',
}

# ── IDX Universe — LQ45 stocks ───────────────────────────────
IDX_UNIVERSE = {
    # ticker: (weight, sector)
    'BBCA.JK': (1.0, 'Banking'),
    'BBRI.JK': (1.0, 'Banking'),
    'BMRI.JK': (1.0, 'Banking'),
    'BBNI.JK': (1.0, 'Banking'),
    'BRIS.JK': (1.0, 'Banking'),
    'TLKM.JK': (1.0, 'Telecom'),
    'EXCL.JK': (1.0, 'Telecom'),
    'ISAT.JK': (1.0, 'Telecom'),
    'UNVR.JK': (1.0, 'Consumer'),
    'ICBP.JK': (1.0, 'Consumer'),
    'INDF.JK': (1.0, 'Consumer'),
    'KLBF.JK': (1.0, 'Consumer'),
    'MYOR.JK': (1.0, 'Consumer'),
    'ASII.JK': (1.0, 'Industrial'),
    'SMGR.JK': (1.0, 'Industrial'),
    'INTP.JK': (1.0, 'Industrial'),
    'ADRO.JK': (1.0, 'Energy'),
    'PTBA.JK': (1.0, 'Energy'),
    'ITMG.JK': (1.0, 'Energy'),
    'PGAS.JK': (1.0, 'Energy'),
    'MEDC.JK': (1.0, 'Energy'),
    'BSDE.JK': (1.0, 'Property'),
    'CTRA.JK': (1.0, 'Property'),
    'PWON.JK': (1.0, 'Property'),
    'SIDO.JK': (1.0, 'Healthcare'),
    'MIKA.JK': (1.0, 'Healthcare'),
    'ADMF.JK': (1.0, 'Finance'),
    'BFIN.JK': (1.0, 'Finance'),
    'GOTO.JK':  (1.0, 'Telecom'),
    'EMTK.JK':  (1.0, 'Telecom'),
    'HMSP.JK':  (1.0, 'Consumer'),
    'GGRM.JK':  (1.0, 'Consumer'),
    'AMRT.JK':  (1.0, 'Consumer'),
    'ACES.JK':  (1.0, 'Consumer'),
    'MAPI.JK':  (1.0, 'Industrial'),
    'SCMA.JK':  (1.0, 'Industrial'),
    'INCO.JK':  (1.0, 'Energy'),
    'HRUM.JK':  (1.0, 'Energy'),
    'MBMA.JK':  (1.0, 'Energy'),
    'JSMR.JK':  (1.0, 'Property'),
    'SMRA.JK':  (1.0, 'Property'),
    'HEAL.JK':  (1.0, 'Healthcare'),
    'BBTN.JK':  (1.0, 'Banking'),
    'ARTO.JK':  (1.0, 'Banking'),
    'BTPS.JK':  (1.0, 'Banking'),
}

MACRO_TICKERS = {
    'IHSG':   '^JKSE',
    'USDIDR': 'IDR=X',
    'GOLD':   'GC=F',
    'VIX':    '^VIX',
    'DXY':    'DX-Y.NYB',
    'COAL':   'MTF=F',
    'NICKEL': 'HG=F',
}
