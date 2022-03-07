import re

emotikony = ":) ;) ;( :> :< ;< :-) ;-)"

znalezione_emotikony = re.findall(r"[;|:][-]?[\)|\(|<|>]", emotikony)
print(znalezione_emotikony)
