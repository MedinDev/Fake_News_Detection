import os
import json
import time
from datetime import datetime

def generate_index(root_dir):
    index_data = {}
    
    # Define categories based on top-level folders
    categories = {
        'drafts': 'Draft',
        'templates': 'Template',
        'research': 'Research',
        'media': 'Media',
        'published-pieces': 'Published',
        'editorial-guidelines': 'Guideline'
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root directory itself to avoid processing index.json if it exists
        if dirpath == root_dir:
            continue
            
        rel_path = os.path.relpath(dirpath, root_dir)
        top_folder = rel_path.split(os.sep)[0]
        
        category = categories.get(top_folder, 'Unknown')
        
        for filename in filenames:
            if filename == 'index.json' or filename.startswith('.'):
                continue
                
            file_path = os.path.join(dirpath, filename)
            rel_file_path = os.path.relpath(file_path, root_dir)
            
            # Get last modified timestamp
            mtime = os.path.getmtime(file_path)
            last_modified = datetime.fromtimestamp(mtime).isoformat()
            
            # Determine status
            status = 'Active'
            if 'drafts' in rel_path:
                status = 'Draft'
            elif 'published-pieces' in rel_path:
                status = 'Published'
            elif 'templates' in rel_path:
                status = 'Template'
            
            index_data[rel_file_path] = {
                'category': category,
                'status': status,
                'last_modified': last_modified
            }

    with open(os.path.join(root_dir, 'index.json'), 'w') as f:
        json.dump(index_data, f, indent=4)
        
    print(f"Index generated with {len(index_data)} entries.")

if __name__ == "__main__":
    # Assumes script is run from project root or checks relative path
    # If run from root, 'article-writing' is the target
    target_dir = 'article-writing'
    if not os.path.exists(target_dir):
        # Fallback if run from inside article-writing
        if os.path.exists('drafts'): # simple check
             target_dir = '.'
        else:
             print("Error: 'article-writing' directory not found.")
             exit(1)
             
    generate_index(os.path.abspath(target_dir))
