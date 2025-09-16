import psutil
import win32api
import os
import wmi
import time
import subprocess
import difflib
import winreg
import win32evtlog
import GPUtil
from pySMART import Device
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

class AISysManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intents = {
            "check_disk": ["check disk space", "how much space is left", "disk usage"],
            "check_cpu": ["check cpu", "cpu usage", "how’s my processor"],
            "kill_process": ["kill process", "stop a program", "end task"],
            "system_info": ["system info", "tell me about my pc", "what’s my setup"],
            "check_memory": ["check memory", "ram usage", "how much ram is free"],
            "check_network": ["check network", "internet status", "network usage"],
            "check_temp": ["check temperature", "cpu temp", "is my pc hot"],
            "help": ["help", "what can you do", "list commands"],
            "check_startup": ["check startup", "startup programs", "what runs at boot"],
            "check_battery": ["check battery", "battery health", "how’s my battery"],
            "check_logs": ["check logs", "event logs", "system warnings"],
            "set_priority": ["set priority", "change process priority", "boost process"],
            "check_gpu": ["check gpu", "gpu usage", "graphics card status"],
            "check_disk_health": ["check disk health", "smart data", "disk status"],
            "run_benchmark": ["run benchmark", "test system", "performance check"],
            "set_power_plan": ["set power plan", "change power mode", "power settings"]
        }
        self.intent_embeddings = {
            intent: self.model.encode(examples, convert_to_tensor=True)
            for intent, examples in self.intents.items()
        }
        self.drive_types = self.detect_drive_types()
        if not os.path.exists("reports"):
            os.makedirs("reports")

    def detect_drive_types(self):
        """Detect SSD vs HDD."""
        drives = {}
        for partition in psutil.disk_partitions():
            drive = partition.device
            try:
                disk_io = psutil.disk_io_counters(perdisk=True).get(drive.split("\\")[0], None)
                if disk_io:
                    is_ssd = disk_io.read_time < 100 and disk_io.write_time < 100
                    drives[drive] = "SSD" if is_ssd else "HDD"
                else:
                    drives[drive] = "Unknown"
            except Exception:
                drives[drive] = "Unknown"
        return drives

    def execute_command(self, command):
        """Run a local command."""
        return os.popen(command).read()

    def classify_intent(self, user_input):
        """Classify user input into an intent."""
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        best_intent, best_score = None, -1
        for intent, embeddings in self.intent_embeddings.items():
            similarity = util.pytorch_cos_sim(user_embedding, embeddings).max().item()
            if similarity > best_score:
                best_score = similarity
                best_intent = intent
        return best_intent if best_score > 0.7 else None

    def check_disk_space(self):
        """Check free disk space with drive type context."""
        drive = "C:\\"
        free_bytes = win32api.GetDiskFreeSpaceEx(drive)[0]
        drive_type = self.drive_types.get(drive, "Unknown")
        advice = (
            "Plenty of space left!" if free_bytes > 10 * (1024**3) else
            "Running low—might want to clean up."
        )
        if drive_type == "SSD":
            advice += " No need to defrag this SSD."
        elif drive_type == "HDD":
            advice += " Consider defragmenting this HDD if it’s slow."
        return f"Free space on {drive} ({drive_type}): {free_bytes / (1024**3):.2f} GB. {advice}"

    def check_cpu_usage(self):
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        advice = (
            "All good here." if cpu_percent < 70 else
            "CPU’s working hard—check Task Manager."
        )
        return f"CPU usage: {cpu_percent}%. {advice}"

    def kill_process(self, process_name=None):
        """Kill a process with fuzzy search and confirmation."""
        if not process_name:
            process_name = input("Enter a process name or part of it: ")
        
        print("Note: Microsoft Store apps (e.g., Hulu) may run under 'msedge.exe' or 'WWAHost.exe' and be hard to detect.")
        search_term = process_name.replace(" ", "").lower()
        
        matches = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                proc_name = proc.info['name'].lower()
                similarity = difflib.SequenceMatcher(None, search_term, proc_name).ratio()
                if similarity > 0.625 or search_term in proc_name:
                    matches.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not matches:
            print(f"No close matches for '{process_name}'. Showing top 5 active processes instead:")
            process_list = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    process_list.append((proc.info['memory_info'].rss, proc.info))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            matches = [info for _, info in sorted(process_list, key=lambda x: x[0], reverse=True)[:5]]
        
        print("Found these processes:")
        for i, match in enumerate(matches):
            print(f"{i}: {match['name']} (PID: {match['pid']}, Memory: {match['memory_info'].rss / (1024**2):.2f} MB)")
        
        choice = input("Enter the number of the process to kill (or 'cancel'): ")
        if choice.lower() == "cancel" or not choice.isdigit() or int(choice) >= len(matches):
            return "Kill canceled."
        
        selected = matches[int(choice)]
        name = selected['name']
        
        desc = {
            "notepad.exe": "a simple text editor",
            "chrome.exe": "Google Chrome web browser",
            "explorer.exe": "Windows File Explorer",
            "cmd.exe": "Command Prompt",
            "wwahost.exe": "a container for Microsoft Store apps",
            "runtimebroker.exe": "a system process for Store app permissions",
            "msedge.exe": "Microsoft Edge browser, may host Store apps or PWAs"
        }.get(name.lower(), "an unknown application")
        
        confirm = input(f"Kill {name} (PID: {selected['pid']}) - {desc}? (yes/no): ")
        if confirm.lower() != "yes":
            return "Kill canceled."
        
        cmd = f"taskkill /PID {selected['pid']} /F"
        result = self.execute_command(cmd)
        return result if result else f"Process {name} (PID: {selected['pid']}) terminated."

    def get_system_info(self):
        """Basic system info."""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        drive_info = ", ".join([f"{d} ({t})" for d, t in self.drive_types.items()])
        return (
            f"System Info:\n"
            f"- CPU Cores: {cpu_count}\n"
            f"- Memory: {memory.total / (1024**3):.2f} GB total, "
            f"{memory.available / (1024**3):.2f} GB free\n"
            f"- Drives: {drive_info}"
        )

    def check_memory_usage(self):
        """Check RAM usage with top memory hogs."""
        memory = psutil.virtual_memory()
        used = memory.used / (1024**3)
        total = memory.total / (1024**3)
        percent = memory.percent
        advice = (
            "Memory looks fine." if percent < 80 else
            "RAM’s almost maxed out—close some apps."
        )
        processes = [(p.info['memory_info'].rss, p.info['name']) for p in psutil.process_iter(['name', 'memory_info'])]
        top_hogs = sorted(processes, reverse=True)[:5]
        hog_list = "\n".join([f"- {name}: {size / (1024**3):.2f} GB" for size, name in top_hogs])
        return (
            f"Memory: {used:.2f}/{total:.2f} GB used ({percent}%). {advice}\n"
            f"Top memory users:\n{hog_list}"
        )

    def check_network_status(self):
        """Check network with ping and real-time bandwidth."""
        print("Running network test—this will take about 5 seconds...")
        net_io_start = psutil.net_io_counters()
        time.sleep(5)
        net_io_end = psutil.net_io_counters()
        sent_rate = (net_io_end.bytes_sent - net_io_start.bytes_sent) / 5 / (1024**2)
        recv_rate = (net_io_end.bytes_recv - net_io_start.bytes_recv) / 5 / (1024**2)
        
        try:
            ping_output = subprocess.run(["ping", "-n", "4", "google.com"], capture_output=True, text=True)
            ping_result = ping_output.stdout
            ping_advice = "Internet looks good." if "time=" in ping_result else "Couldn’t reach Google—check your connection."
        except Exception:
            ping_result = "Ping failed."
            ping_advice = "Network issue detected—modem or router might be down."

        return (
            f"Network Bandwidth (5s sample): Sent {sent_rate:.2f} MB/s, Received {recv_rate:.2f} MB/s\n"
            f"Ping to Google:\n{ping_result}\n{ping_advice}"
        )

    def check_system_temp(self):
        """Check CPU temperature using WMI."""
        try:
            w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            sensors = w.Sensor()
            cpu_temp = None
            for sensor in sensors:
                if sensor.SensorType == "Temperature" and "CPU" in sensor.Name:
                    cpu_temp = sensor.Value
                    break
            if cpu_temp:
                advice = (
                    "Temp looks normal." if cpu_temp < 80 else
                    "CPU’s hot—check cooling or reduce load."
                )
                return f"CPU Temperature: {cpu_temp}°C. {advice}"
            else:
                return "No CPU temp found. Download OpenHardwareMonitor from openhardwaremonitor.org, run it, then try again."
        except Exception:
            return "Temperature check failed. Ensure OpenHardwareMonitor is installed and running (get it from openhardwaremonitor.org)."

    def check_startup(self):
        """Check startup programs."""
        startup_items = []
        reg_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run")
        ]
        
        for hive, path in reg_paths:
            try:
                key = winreg.OpenKey(hive, path, 0, winreg.KEY_READ)
                for i in range(winreg.QueryInfoKey(key)[1]):
                    name, value, _ = winreg.EnumValue(key, i)
                    startup_items.append(f"- {name}: {value}")
                winreg.CloseKey(key)
            except WindowsError:
                continue
        
        if not startup_items:
            return "No startup programs found in common registry locations."
        
        return (
            "Startup Programs:\n" +
            "\n".join(startup_items) +
            "\n\nThese run automatically at boot. Too many can slow startup—check Task Manager’s Startup tab to disable."
        )

    def check_battery(self):
        """Check battery health."""
        report_path = os.path.join("reports", "battery_report.html")
        os.system(f"powercfg /batteryreport /output \"{report_path}\"")
        time.sleep(2)
        
        if not os.path.exists(report_path):
            return "Battery report generation failed. Are you on a laptop with a battery?"
        
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            design_capacity = None
            full_charge_capacity = None
            for tr in soup.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    if "DESIGN CAPACITY" in tds[0].text:
                        capacity_text = tds[1].text.strip().replace(',', '')
                        design_capacity = int("".join(filter(str.isdigit, capacity_text))) if capacity_text else None
                    elif "FULL CHARGE CAPACITY" in tds[0].text:
                        capacity_text = tds[1].text.strip().replace(',', '')
                        full_charge_capacity = int("".join(filter(str.isdigit, capacity_text))) if capacity_text else None
            
            if design_capacity and full_charge_capacity:
                health_percent = (full_charge_capacity / design_capacity) * 100
                advice = (
                    "Battery’s in good shape." if health_percent > 80 else
                    "Battery health is degrading—consider replacing it soon."
                )
                return (
                    f"Battery Health:\n"
                    f"- Design Capacity: {design_capacity} mWh\n"
                    f"- Full Charge Capacity: {full_charge_capacity} mWh\n"
                    f"- Health: {health_percent:.1f}%\n"
                    f"{advice}"
                )
            else:
                return "Couldn’t find battery capacity data in report. Check 'reports/battery_report.html' manually."
        except Exception as e:
            return f"Error parsing battery report: {str(e)}"
        finally:
            pass

    def check_logs(self):
        """Check and analyze event logs with user input."""
        print("Let’s refine your event log search...")
        
        days = input("How many days back to search (1-30, default 7)? ")
        try:
            days = min(max(int(days), 1), 30) if days.strip() else 7
        except ValueError:
            days = 7
        
        level_input = input("Filter by level (1=Critical, 2=Error, 4=Warning, e.g., '1,2', default 0=Unsure)? ").strip()
        level_map = {"1": 1, "2": 2, "4": 4}
        levels = set()
        if level_input:
            for part in level_input.split(','):
                if part.strip() in level_map:
                    levels.add(level_map[part.strip()])
        if not levels:
            levels = {1, 2, 4}
        
        keyword = input("Enter a keyword to search (e.g., 'dll', optional): ").strip().lower()
        error_code = input("Enter an error code (e.g., '0x80070491', optional): ").strip().lower()
        
        print(f"Searching logs from last {days} days—this may take a moment...")
        cutoff = datetime.now() - timedelta(days=days)
        logs = {"Application": [], "System": []}
        
        for log_type in logs.keys():
            try:
                hand = win32evtlog.OpenEventLog(None, log_type)
                flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                events = win32evtlog.ReadEventLog(hand, flags, 0)
                
                while events:
                    for event in events:
                        event_time = event.TimeGenerated
                        if event_time < cutoff:
                            break
                        level_num = event.EventType
                        if level_num not in levels:
                            continue
                        
                        level_str = {1: "Critical", 2: "Error", 4: "Warning"}.get(level_num, "Other")
                        source = event.SourceName
                        desc = str(event.StringInserts) if event.StringInserts else "No description."
                        details = f"Event ID: {event.EventID & 0xFFFF} | Category: {event.EventCategory}"
                        
                        text = (desc + " " + details).lower()
                        if (not keyword or difflib.SequenceMatcher(None, keyword, text).ratio() > 0.625 or keyword in text) and \
                           (not error_code or error_code in text):
                            friendly_desc = self._synthesize_log(source, desc, details)
                            logs[log_type].append((event_time, level_str, source, friendly_desc, details))
                    
                    events = win32evtlog.ReadEventLog(hand, flags, 0)
                win32evtlog.CloseEventLog(hand)
            except Exception as e:
                print(f"Error reading {log_type} log: {str(e)}")
                continue
        
        output = []
        error_count = sum(1 for log in logs.values() for _, level, _, _, _ in log if level in ["Critical", "Error"])
        warning_count = sum(1 for log in logs.values() for _, level, _, _, _ in log if level == "Warning")
        
        for log_type, events in logs.items():
            if events:
                output.append(f"\n{log_type} Log (Last {days} Days):")
                for time, level, source, friendly_desc, details in sorted(events, key=lambda x: x[0], reverse=True)[:5]:
                    output.append(f"- {time} | {level} | {source}: {friendly_desc} ({details})")
        
        if not any(logs.values()):
            return f"No matching events found in the last {days} days."
        
        analysis = [
            f"\nSummary (Last {days} Days):",
            f"- Total Errors/Critical: {error_count}",
            f"- Total Warnings: {warning_count}"
        ]
        if error_count > 5:
            analysis.append("Several serious issues detected—check hardware or software problems.")
        elif warning_count > 10:
            analysis.append("Lots of warnings—your system might be unstable; look into frequent issues.")
        else:
            analysis.append("Things look mostly stable.")
        
        return "\n".join(output + analysis)

    def _synthesize_log(self, source, desc, details):
        """Rephrase log entries into user-friendly terms."""
        known_issues = {
            "Application Error": lambda d: f"An app ({d.split('Event ID')[0].strip()}) crashed unexpectedly." if "crashed" not in d.lower() else d,
            "Windows Update": lambda d: "Windows Update hit a snag—might need a restart or manual update.",
            "Service Control Manager": lambda d: "A background service failed to start properly.",
            "DLL": lambda d: "A system file (DLL) didn’t load right—could be a missing or corrupt file."
        }
        
        for key, fn in known_issues.items():
            if key.lower() in source.lower() or key.lower() in desc.lower():
                return fn(desc)
        
        if "failed" in desc.lower():
            return f"Something ({source}) didn’t work as expected."
        elif "warning" in desc.lower():
            return f"System flagged a potential issue with {source}."
        return desc

    def set_process_priority(self, process_name=None):
        """Set priority of a running process."""
        if not process_name:
            process_name = input("Enter a process name or part of it: ")
        
        search_term = process_name.replace(" ", "").lower()
        matches = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if search_term in proc.info['name'].lower():
                    matches.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not matches:
            return f"No process found matching '{process_name}'."
        
        print("Found these processes:")
        for i, proc in enumerate(matches):
            print(f"{i}: {proc.info['name']} (PID: {proc.info['pid']})")
        
        choice = input("Enter the number of the process to adjust (or 'cancel'): ")
        if choice.lower() == "cancel" or not choice.isdigit() or int(choice) >= len(matches):
            return "Priority adjustment canceled."
        
        selected = matches[int(choice)]
        priority_map = {
            "low": psutil.BELOW_NORMAL_PRIORITY_CLASS,
            "normal": psutil.NORMAL_PRIORITY_CLASS,
            "high": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            "realtime": psutil.REALTIME_PRIORITY_CLASS
        }
        
        priority = input("Set priority (low, normal, high, realtime): ").lower()
        if priority not in priority_map:
            return "Invalid priority. Use: low, normal, high, realtime."
        
        try:
            selected.nice(priority_map[priority])
            return f"Set {selected.info['name']} (PID: {selected.info['pid']}) to {priority} priority."
        except psutil.AccessDenied:
            return "Access denied—run as admin to change priority."

    def check_gpu(self):
        """Check GPU usage, temperature, and memory."""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return "No GPU detected or GPUtil failed to initialize."
            
            output = []
            for gpu in gpus:
                output.append(
                    f"GPU {gpu.id}: {gpu.name}\n"
                    f"- Usage: {gpu.load * 100:.1f}%\n"
                    f"- Temperature: {gpu.temperature}°C\n"
                    f"- Memory: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} MB "
                    f"({gpu.memoryUtil * 100:.1f}% used)"
                )
                if gpu.temperature > 85:
                    output.append("Warning: GPU is running hot—check cooling!")
            return "\n".join(output)
        except Exception as e:
            return f"Error checking GPU: {str(e)}. Ensure GPU drivers are installed."

    def check_disk_health(self):
        """Check SMART data for disk health."""
        try:
            output = []
            for partition in psutil.disk_partitions():
                drive = partition.device.split("\\")[0]  # e.g., "C:"
                disk = Device(drive)
                if disk.smart_enabled:
                    health = disk.assessment if disk.assessment else "Unknown"
                    temp = disk.attributes[194].raw if 194 in disk.attributes else "N/A"
                    output.append(
                        f"Drive {drive} ({disk.model}):\n"
                        f"- Health: {health}\n"
                        f"- Temperature: {temp}°C\n"
                        f"- Type: {disk.interface}"
                    )
                else:
                    output.append(f"Drive {drive}: SMART not supported.")
            return "\n".join(output) if output else "No SMART-capable drives found."
        except Exception as e:
            return f"Error checking disk health: {str(e)}. Run as admin for SMART data."

    def run_benchmark(self):
        """Run a simple CPU, memory, and disk benchmark."""
        print("Running benchmark—this will take about 10 seconds...")
        
        # CPU: Simple arithmetic test
        start = time.time()
        for _ in range(1000000):
            _ = 12345 * 67890
        cpu_time = time.time() - start
        cpu_score = 1000000 / cpu_time / 1000  # Ops per second / 1000
        
        # Memory: Write/read test
        data = bytearray(1024 * 1024 * 100)  # 100 MB
        start = time.time()
        _ = data[:]
        mem_time = time.time() - start
        mem_score = 100 / mem_time  # MB/s
        
        # Disk: Write/read test
        test_file = "reports/benchmark_test.bin"
        with open(test_file, "wb") as f:
            start = time.time()
            f.write(data)
        disk_write_time = time.time() - start
        with open(test_file, "rb") as f:
            start = time.time()
            _ = f.read()
        disk_read_time = time.time() - start
        disk_score = 100 / (disk_write_time + disk_read_time)  # MB/s
        os.remove(test_file)
        
        return (
            f"Benchmark Results:\n"
            f"- CPU: {cpu_score:.1f} kOps/s\n"
            f"- Memory: {mem_score:.1f} MB/s\n"
            f"- Disk: {disk_score:.1f} MB/s (write + read)"
        )

    def set_power_plan(self):
        """List and set Windows power plans."""
        result = subprocess.run("powercfg /list", capture_output=True, text=True)
        plans = {}
        for line in result.stdout.splitlines():
            if "GUID" in line:
                guid = line.split()[3]
                name = " ".join(line.split()[4:]).strip("*")
                plans[name] = guid
        
        if not plans:
            return "No power plans found."
        
        print("Available power plans:")
        for i, (name, guid) in enumerate(plans.items()):
            print(f"{i}: {name} ({guid})")
        
        choice = input("Enter the number of the power plan to activate (or 'cancel'): ")
        if choice.lower() == "cancel" or not choice.isdigit() or int(choice) >= len(plans):
            return "Power plan change canceled."
        
        selected_name = list(plans.keys())[int(choice)]
        selected_guid = plans[selected_name]
        subprocess.run(f"powercfg /setactive {selected_guid}", shell=True)
        return f"Activated power plan: {selected_name}"

    def help(self):
        """List available commands."""
        return (
            "Here’s what I can do:\n"
            "- 'check disk space': See free space on your C: drive.\n"
            "- 'check cpu': Check CPU usage.\n"
            "- 'kill process': Stop a running program.\n"
            "- 'system info': Get details about your PC.\n"
            "- 'check memory': See RAM usage and top users.\n"
            "- 'check network': Check bandwidth and ping Google.\n"
            "- 'check temperature': Get CPU temp (needs OpenHardwareMonitor).\n"
            "- 'check startup': List programs that run at boot.\n"
            "- 'check battery': Check battery health (laptops only).\n"
            "- 'check logs': Analyze recent event log warnings and errors.\n"
            "- 'set priority': Adjust a process’s CPU priority.\n"
            "- 'check gpu': Monitor GPU usage, temp, and memory.\n"
            "- 'check disk health': Check SMART data for disk health.\n"
            "- 'run benchmark': Test CPU, memory, and disk performance.\n"
            "- 'set power plan': Switch Windows power modes.\n"
            "- 'help': Show this list."
        )

    def run(self):
        """Main loop."""
        print("AI Sys Manager ready. Ask me anything about your system! (Type 'exit' to quit)")
        while True:
            user_input = input("What do you need? ")
            if user_input.lower() == "exit":
                break
            intent = self.classify_intent(user_input)
            if intent == "check_disk":
                print(self.check_disk_space())
            elif intent == "check_cpu":
                print(self.check_cpu_usage())
            elif intent == "kill_process":
                print(self.kill_process())
            elif intent == "system_info":
                print(self.get_system_info())
            elif intent == "check_memory":
                print(self.check_memory_usage())
            elif intent == "check_network":
                print(self.check_network_status())
            elif intent == "check_temp":
                print(self.check_system_temp())
            elif intent == "check_startup":
                print(self.check_startup())
            elif intent == "check_battery":
                print(self.check_battery())
            elif intent == "check_logs":
                print(self.check_logs())
            elif intent == "set_priority":
                print(self.set_process_priority())
            elif intent == "check_gpu":
                print(self.check_gpu())
            elif intent == "check_disk_health":
                print(self.check_disk_health())
            elif intent == "run_benchmark":
                print(self.run_benchmark())
            elif intent == "set_power_plan":
                print(self.set_power_plan())
            elif intent == "help":
                print(self.help())
            else:
                print("Not sure what you mean. Say 'help' for a list of commands!")

if __name__ == "__main__":
    ai = AISysManager()
    ai.run()