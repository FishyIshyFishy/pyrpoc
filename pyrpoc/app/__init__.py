"""The composition root: builds every part, connects them, and shows the window."""

# The only package that may import everything, so it is obvious when it grows. Inside, each
# imports only what is below it: session, window, menubar, then theme and application.
