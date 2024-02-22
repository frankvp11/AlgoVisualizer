
from nicegui import ui



def add(page="main"):
    pages = ["main", "about"]
    icons = ['home', 'info']
    texts = ["Main", "About"]
    targets = ["/", "/about"]
    
    left_drawer = (
        ui.left_drawer(fixed=False, elevated=True)
        .classes("bg-blue-100")
        .props("overlay")
        .style("position:relative;")
        .props("behavior=mobile")
    )

    left_drawer.toggle()



    with left_drawer:
        for i in range(len(pages)):
            with ui.element("div").style("margin-top: 10px; "):
                with ui.link(target=targets[i]).style("text-decoration:none; "):
                    if page == pages[i]:
                        dashboard_row = (
                            ui.row()
                            .style(
                                "border-radius: 20px; margin-top: 5px; margin-bottom: 5px;"
                            )
                            .classes("bg-slate-300")
                        )
                    else:
                        dashboard_row = (
                            ui.row()
                            .style(
                                "border-radius: 20px; margin-top: 5px; margin-bottom: 5px;"
                            )
                            .classes("hover:bg-slate-300")
                        )
                    with dashboard_row:
                        ui.icon(icons[i], size="36px").style(
                            "margin-left: 10px; text-align:center; display:block; "
                        )
                        ui.link(text=texts[i], target=targets[i]).style(
                            "text-decoration:none; color:black;  margin-right: 10px; font-size:20px; display:block; margin-top: 1%;"
                        )



    return left_drawer