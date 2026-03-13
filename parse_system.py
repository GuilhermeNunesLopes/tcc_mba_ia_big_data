import re
import os 
import pandas as pd
import sys 

def automatic_parse(file):
    
    # Define regex patterns for different log formats
    pattern = r'^\[([^\]]+)\]\s+\[([^\]]+)\]\s+(.*)$' #APACHE
    pattern2 = r'^([A-Za-z]{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2})\s+[\w+\:]+\s+([^\[\]]+)\[.*?\]: (.*)$' #SYSLOG: group1=date/time, group2=process (e.g. sshd(pam_unix)), group3=message
    pattern3 = r'^(\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+(\w+)\s+([^:]+):\s+(.*)$' #SPARK

      

    data = []

    with open(file, 'r') as f:
        for line in f:
            # Try matching each pattern in order
            m = re.match(pattern, line)
            # If pattern matches, extract date/time, level, source, and event
            if m:
                date_time = m.group(1)
                level = m.group(2)
                source = 'apache'
                event = m.group(3)
                data.append({'Date': date_time, 'Level': level, 'Source': source, 'Event': event})
            # If first pattern doesn't match, try the second pattern
            else:
                m2 = re.match(pattern2, line)
                if m2:
                    date_time = m2.group(1)
                    level = 'info'
                    source = m2.group(2)
                    event = m2.group(3)
                    data.append({'Date': date_time, 'Level': level, 'Source': source, 'Event': event})
                else:
                    m3 = re.match(pattern3, line)
                    # If third pattern doesn't match, skip the line
                    if m3:
                        date_time = m3.group(1)
                        level = m3.group(2)
                        source = m3.group(3)
                        event = m3.group(4)
                        data.append({'Date': date_time, 'Level': level, 'Source': source, 'Event': event})
                    # else: skip line
        if not data:
            print("No matches found in the log file.")
            return pd.DataFrame()  # Return empty DataFrame if no matches
    
        print(f"Found {len(data)} matches.")
        df = pd.DataFrame(data)
          
    return df
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_system.py <log_file>")
        sys.exit(1)
    log_file = sys.argv[1]
    if not os.path.isfile(log_file):
        print(f"Error: File '{log_file}' does not exist.")
        sys.exit(1)
    df = automatic_parse(log_file)
    print(df)