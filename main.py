from nicegui import ui, app, APIRouter
import cards


app.add_static_files("/static", 'static')

head_html = '''
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script type="text/javascript" id="MathJax-script" async
              src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
            </script>
            '''
ui.add_head_html(head_html)


header = ui.header()
cards.add()


ui.run()
