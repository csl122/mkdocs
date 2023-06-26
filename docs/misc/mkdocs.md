# References for MkDocs

## Buttons

### Default Button
[Subscribe to our newsletter](#){ .md-button }

### Primary Button
[Subscribe to our newsletter](#){ .md-button .md-button--primary }

### Icon Button
[Send :fontawesome-solid-paper-plane:](#){ .md-button }


## Content Tabs
=== "Unordered list"

    * Sed sagittis eleifend rutrum
    * Donec vitae suscipit est
    * Nulla tempor lobortis orci

=== "Ordered list"

    1. Sed sagittis eleifend rutrum
    2. Donec vitae suscipit est
    3. Nulla tempor lobortis orci


## Admonitions

### Note
!!! note

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

### Collapsible blocks
??? note

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

???+ note

    Adding a + after the ??? token renders the block expanded.

### Supported admonitions

-  `note`
-  `warning`
-  `tip`
-  `abstract`
-  `info`
-  `success`
-  `question`
-  `failure`
-  `danger`
-  `bug`
-  `example`
-  `quote`

## Code Blocks

### Python Block

``` py
import torch 
```

### Python Block with Title

``` py title="test.py"
import numpy as np
```

### Python Block with Line Numbers

``` py linenums="1"
import torch 
import numpy as np

def to_be_tensor(np_arr):
    return torch.from_numpy(np_arr)

```

### Python Block with Highlighted Lines

``` py hl_lines="1 2"
import torch 
import numpy as np

def to_be_tensor(np_arr):
    return torch.from_numpy(np_arr)

```

### Adding Annotations & Stripping Line Noise

``` yaml
theme:
  features:
    - content.code.annotate # (1)

Following line strips line noise from the code block
# (2)!
```

1.  :man_raising_hand: I'm a code annotation! I can contain `code`, __formatted
    text__, images, ... basically anything that can be written in Markdown.
2.  Look ma, less line noise!

### Embedding External Code

``` title="Code block with external content"
--8<-- "mkdocs.yml"
```

### Footnote

Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^2]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
