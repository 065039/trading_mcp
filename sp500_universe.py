"""Bundled S&P 500 symbols used for deterministic two-day scan rotation."""

import hashlib
from datetime import date, datetime, timedelta, timezone

SP500_TICKERS = tuple(
    """
    MMM AOS ABT ABBV ACN ADBE AMD AES AFL A APD ABNB AKAM ALB ARE ALGN
    ALLE LNT ALL GOOGL GOOG MO AMZN AMCR AEE AEP AXP AIG AMT AWK AMP AME
    AMGN APH ADI AON APA APO AAPL AMAT APP APTV ACGL ADM ARES ANET AJG AIZ
    T ATO ADSK ADP AZO AVY AXON BKR BALL BAC BAX BDX BRK.B BBY TECH BIIB
    BLK BX XYZ BE BNY BA BKNG BSX BMY AVGO BR BRO BF.B BG BXP CHRW CDNS CPT
    COF CAH CCL CARR CVNA CASY CAT CBOE CBRE CDW COR CNC CNP CF CRL SCHW
    CHTR CVX CMG CB CHD CIEN CI CINF CTAS CSCO C CFG CLX CME CMS KO CTSH
    COHR COIN CL CMCSA FIX COP ED STZ CEG COO CPRT GLW CPAY CSGP COST CRH
    CRWD CCI CSX CMI CVS DHR DRI DDOG DVA DECK DE DELL DAL DVN DXCM FANG
    DLR DG DLTR D DPZ DASH DOV DOW DHI DTE DUK DD ETN EBAY ECHO ECL EIX EW
    ELV EME EMR ETR EOG EQT EFX EQIX ERIE ESS EL EG EVRG P ES EXC EXE EXPE
    EXPD EXR XOM FFIV FDS FICO FAST FRT FDX FDXF FERG FIS FITB FSLR FE
    FISV FLEX F FTNT FTV FOXA FOX BEN FCX GRMN IT GE GEHC GEV GEN GNRC GD
    GIS GM GPC GILD GPN GL GDDY GS HAL HIG HAS HCA DOC HSIC HSY HPE HLT HD
    HONA HON HRL HST HWM HPQ HUBB HUM HBAN HII IBM IEX IDXX ITW ILMN INCY
    IR PODD INTC IBKR ICE IFF IP INTU ISRG IVZ INVH IQV IRM JBHT JBL JKHY
    J JNJ JCI JPM KVUE KDP KEY KEYS KMB KIM KMI KKR KLAC KHC KR LHX LH
    LRCX LVS LDOS LEN LII LLY LIN LYV LMT L LOW LULU LITE LYB MTB MPC MAR
    MRSH MLM MRVL MAS MA MKC MCD MCK MDT MRK META MET MTD MGM MCHP MU MSFT
    MAA MRNA MDLZ MPWR MNST MCO MS MOS MSI MSCI NDAQ NTAP NFLX NEM NWSA NWS
    NEE NKE NI NDSN NSC NTRS NOC NCLH NRG NUE NVDA NVR NXPI ORLY OXY ODFL
    OMC ON OKE ORCL OTIS PCAR PKG PLTR PANW PH PAYX PYPL PNR PEP PFE PCG
    PM PSX PNW PNC PPG PPL PFG PG PGR PLD PRU PEG PTC PSA PHM PWR QCOM DGX
    Q RL RJF RDDT RTX O REG REGN RF RSG RMD RVTY HOOD ROK ROL ROP ROST RCL
    SPGI CRM SNDK SBAC SLB STX SRE NOW SHW SPG SKYD SWKS SJM SW SNA SOLV
    SO LUV SWK SBUX STT STLD STE SYK SMCI SYF SNPS SYY TMUS TROW TTWO TPR
    TRGP TGT TEL TDY TER TSLA TXN TPL TXT TMO TJX TKO TSCO TT TDG TRV TRMB
    TFC TWLO TYL TSN USB UBER UDR ULTA UNP UAL UPS URI UNH UHS VLO VEEV
    VTR VLTO VRSN VRSK VZ VRTX VRT VTRS VICI V VST VMRK VMC VYLR WRB GWW
    WAB WMT DIS WM WAT WEC WFC WELL WST WDC WY WSM WMB WTW WDAY WYNN XEL
    XYL YUM ZBRA ZBH ZTS
    """.split()
)

SCAN_CANDIDATE_COUNT = 60


def get_rotation_period_start(current_time: datetime | None = None) -> date:
    """Return the start date of the current fixed 48-hour UTC rotation period."""
    current_time = current_time or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    current_date = current_time.astimezone(timezone.utc).date()
    epoch = date(1970, 1, 1)
    period_index = (current_date - epoch).days // 2
    return epoch + timedelta(days=period_index * 2)


def get_rotating_candidates(
    current_time: datetime | None = None,
) -> tuple[str, ...]:
    """Return a deterministic, pseudo-random 60-symbol slice that changes every two days."""
    period_start = get_rotation_period_start(current_time)
    period_index = (period_start - date(1970, 1, 1)).days // 2
    ranked_symbols = sorted(
        SP500_TICKERS,
        key=lambda symbol: hashlib.sha256(symbol.encode("utf-8")).digest(),
    )
    start = (period_index * SCAN_CANDIDATE_COUNT) % len(ranked_symbols)
    rotated_symbols = ranked_symbols[start:] + ranked_symbols[:start]
    return tuple(rotated_symbols[:SCAN_CANDIDATE_COUNT])
