#!/usr/bin/env python3
"""
FPL Platform - Entry point

Launches the FPL Platform web application
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fpl_platform.app import create_app

if __name__ == '__main__':
    # Get configuration from environment
    config_name = os.environ.get('FLASK_ENV', 'development')
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '127.0.0.1')
    
    # Create app
    app = create_app(config_name)
    
    # Run
    print(f"""
===============================================================
                   FPL PLATFORM v1.0                       
         Fantasy Premier League Bot Platform               
===============================================================
                                                           
  Server starting on: http://{host}:{port}
  Environment: {config_name.upper()}
                                                           
  Features:                                                
    - User Authentication                                  
    - Multi-Team Management                                
    - AI-Powered Bot Integration                           
    - 7-Week Strategic Planning                            
    - Analysis History Tracking                            
                                                           
  Press Ctrl+C to stop the server                          
===============================================================
    """)
    
    app.run(
        host=host,
        port=port,
        debug=(config_name == 'development')
    )

