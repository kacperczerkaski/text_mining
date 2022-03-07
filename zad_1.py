import re

liczby = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"

usuwanie_liczb = re.sub("[0-9]+", "", liczby)
print(usuwanie_liczb)

znaki_html = "<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>"

usuwanie_html = re.sub(r"<.*?>", "", znaki_html)
print(usuwanie_html)

interpunkcja = "Lorem ipsum dolor sit amet, consectetur; adipiscing elit. Sed eget mattis sem. Mauris egestas erat " \
               "quam, ut faucibus eros congue et. In blandit, mi eu porta; lobortis, tortor nisl facilisis leo, at " \
               "tristique augue risus eu risus."

usuwanie_interpunkcji = re.sub("[.;,]+", "", interpunkcja)
print(usuwanie_interpunkcji)
