from nicegui import ui

def display_counter(counter):
    print(counter[0])

count = [0]  # Using a list to store the count

def update_counter():
    count[0] += 1
    display_counter(count)

ui.timer(1, update_counter)

ui.run()
