# WarThunder-tanks-AB-canon-auto-calibration
A python script makes AB tanks can work as well as RB tanks with laser rangefinder.
All algorithm in this script is pure driven by CV (screenshot then ai process) and no memory access needed.

## Requirement

```python
# Python 3.9.23
import re
import time
import cv2
import keyboard
import numpy as np
import pyautogui
import random
from paddleocr import PaddleOCR
from PIL import Image
import json
```



## Usage

#### Config.json Example

```json
{

  "x1y1" : [720, 360],  // Screenshot Rectangle Coord Left Top

  "x2y2" : [1200, 700], // Screenshot Rectangle Coord Right Bottom

  "max_det_num" : 5,	// Detect times (Default 5)

  "min_cali_distance" : 101, // Min detect distance (m)

  "alpha" : 50, // (equal to WT mouse wheel multiplier), Default 50

  "safe_mode" : "False", // Double check before calibration, may cause less efficiency but high Acc (Default False)

  "debug_mode" : "False", // Output screenshot information, allow to alter screenshot position

  "load_path" : "./load.json" // The path of custom tanks config file 

}
```

#### load.json Example

You can add multi round's file in this .json

```json
[

{

  "name": "120L55_dm63",	// name is just a custom tag, name it whatever.

  "path": "./Ballistic/it_leopard_2a7_hungary/dm53.txt"	// the path of the round's data file

},

{

  "name": "120L55_he",

  "path": "./Ballistic/it_leopard_2a7_hungary/dm_11.txt"

}

]
```

