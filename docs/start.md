# Start From Here

## Install

```py title="test.py" linenums="1"
pip install mkdocs
import mkdocs

def abc():
    pass
```

```py hl_lines="1 2", title="test.py"
pip install mkdocs
import mkdocs

def abc():
    pass
```

``` yaml
theme:
  features:
    - content.code.annotate # (1)
```

1.  :man_raising_hand: I'm a code annotation! I can contain `code`, __formatted
    text__, images, ... basically anything that can be written in Markdown.


``` yaml
# (1)!
```

1.  Look ma, less line noise!
