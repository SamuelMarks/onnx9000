"""Module docstring."""

import enum


class DataType(enum.IntEnum):
    """DataType class."""

    kFLOAT = 0
    kHALF = 1
    kINT8 = 2
    kINT32 = 3
    kBOOL = 4
    kUINT8 = 5
    kFP8 = 6
    # Int64 isn't officially supported in TRT in older versions, we must emulate


class ElementWiseOperation(enum.IntEnum):
    """ElementWiseOperation class."""

    kSUM = 0
    kPROD = 1
    kMAX = 2
    kMIN = 3
    kSUB = 4
    kDIV = 5
    kMINIMUM = 6
    kPOW = 7
    kFLOOR_DIV = 8
    kAND = 9
    kOR = 10
    kXOR = 11
    kEQUAL = 12
    kGREATER = 13
    kLESS = 14


class PoolingType(enum.IntEnum):
    """PoolingType class."""

    kMAX = 0
    kAVERAGE = 1
    kMAX_AVERAGE_BLEND = 2


class ActivationType(enum.IntEnum):
    """ActivationType class."""

    kRELU = 0
    kSIGMOID = 1
    kTANH = 2
    kLEAKY_RELU = 3
    kELU = 4
    kSELU = 5
    kSOFTSIGN = 6
    kSOFTPLUS = 7
    kCLIP = 8
    kHARD_SIGMOID = 9
    kSCALED_TANH = 10
    kTHRESHOLDED_RELU = 11


class ScaleMode(enum.IntEnum):
    """ScaleMode class."""

    kUNIFORM = 0
    kCHANNEL = 1
    kELEMENTWISE = 2


class UnaryOperation(enum.IntEnum):
    """UnaryOperation class."""

    kEXP = 0
    kLOG = 1
    kSQRT = 2
    kRECIP = 3
    kABS = 4
    kNEG = 5
    kSIN = 6
    kCOS = 7
    kTAN = 8
    kSINH = 9
    kCOSH = 10
    kASIN = 11
    kACOS = 12
    kATAN = 13
    kASINH = 14
    kACOSH = 15
    kATANH = 16
    kCEIL = 17
    kFLOOR = 18
    kERF = 19
    kNOT = 20
    kSIGN = 21
    kROUND = 22
    kISINF = 23


class ReduceOperation(enum.IntEnum):
    """ReduceOperation class."""

    kSUM = 0
    kPROD = 1
    kMAX = 2
    kMIN = 3
    kAVG = 4


class MatrixOperation(enum.IntEnum):
    """MatrixOperation class."""

    kNONE = 0
    kTRANSPOSE = 1
    kVECTOR = 2


class TopKOperation(enum.IntEnum):
    """TopKOperation class."""

    kMAX = 0
    kMIN = 1


class MemoryPoolType(enum.IntEnum):
    """MemoryPoolType class."""

    kWORKSPACE = 0
    kDLA_MANAGED_SRAM = 1
    kDLA_LOCAL_DRAM = 2
    kDLA_GLOBAL_DRAM = 3


class OptProfileSelector(enum.IntEnum):
    """OptProfileSelector class."""

    kMIN = 0
    kOPT = 1
    kMAX = 2


class BuilderFlag(enum.IntEnum):
    """BuilderFlag class."""

    kFP16 = 0
    kINT8 = 1
    kDEBUG = 2
    kGPU_FALLBACK = 3
    kSTRICT_TYPES = 4
    kREFIT = 5
    kDISABLE_TIMING_CACHE = 6
    kTF32 = 7
    kSPARSE_WEIGHTS = 8
    kSAFETY_SCOPE = 9
    kOBEY_PRECISION_CONSTRAINTS = 10
    kREJECT_EMPTY_ALGORITHMS = 11
    kDIRECT_IO = 12
    kREJECT_EMPTY_TUNING = 13
    kPREFER_PRECISION_CONSTRAINTS = 14
    kFP8 = 15
    kERROR_ON_TIMING_CACHE_MISS = 16
