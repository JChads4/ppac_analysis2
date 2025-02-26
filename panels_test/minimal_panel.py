#!/usr/bin/env python3

import sys
import traceback

# Ensure Panel is imported
try:
    import panel as pn
except ImportError as e:
    print(f"Failed to import Panel: {e}")
    sys.exit(1)

# Explicitly enable Panel extension
pn.extension(template='bootstrap')

def create_app():
    """
    Create a simple Panel application
    """
    # Create a column with various widgets
    return pn.Column(
        pn.pane.Markdown("# SHREC Data Explorer"),
        pn.pane.Markdown("## Interactive Visualization"),
        pn.widgets.Button(name="Click Me"),
        pn.widgets.TextInput(name="Enter text"),
        height=400, 
        width=600
    )

def main():
    print("Starting Panel Application")
    print("=" * 30)
    
    try:
        # Create the app
        app = create_app()
        
        print("App created successfully")
        print("Attempting to serve...")
        
        # Use an explicit import to avoid any scoping issues
        import panel as serving_pn
        
        # Serve the app
        serving_pn.serve(app, port=5007, show=True, verbose=True)
    
    except Exception as e:
        print("ERROR: Failed to start Panel application")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()