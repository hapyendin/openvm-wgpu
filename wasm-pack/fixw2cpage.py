import pathlib
p = pathlib.Path("mkpage_w2c.py")
s = p.read_text()
needle = "log('W2C_GOLD_SHA '+key+': '+(g?'PASS':'FAIL'));"
add = needle + " log('W2C_ROW_SHA '+key+' '+gs[1]+' GOLD '+GOLD[key]);"
assert needle in s and "W2C_ROW_SHA" not in s
p.write_text(s.replace(needle, add))
print("patched")
