import random
from openai import AsyncAzureOpenAI as AzureOpenAI
import typing as t
import os
import httpx
import json
import asyncio
import pandas as pd
import matplotlib.pyplot as plt

class OpenAIClient:
    api_configurations: t.List[t.Dict[str, str]] = [
        {"endpoint": os.environ.get("OPENAI_API_BASE_URL", ""), "api_key": os.environ.get("OPENAI_API_KEY", "")},
        {"endpoint": os.environ.get("SECOND_OPENAI_API_BASE_URL", ""), "api_key": os.environ.get("SECOND_OPENAI_API_KEY", "")}
    ]
    api_version: str = os.environ.get("OPENAI_API_VERSION", "2023-12-01-preview")
    engine_name_llm: str = os.environ.get("OPENAI_API_GPT4_OMNI_128K_20240513", "")
    http_client: httpx.Client = httpx.AsyncClient()
    max_retries: int = int(os.environ.get("MAX_RETRIES", 3))
    
    @classmethod
    async def get_openai_client(cls, api_key: str, endpoint: str) -> AzureOpenAI:
        """Class method to get or create an AzureOpenAI client with class-level configuration."""
        return AzureOpenAI(
            api_key=api_key,
            api_version=cls.api_version,
            azure_endpoint=endpoint,
            http_client=cls.http_client,
            timeout=httpx.Timeout(
                connect=float(os.environ.get("CONNECT_TIMEOUT", 2.0)),
                read=float(os.environ.get("READ_TIMEOUT", 20.0)),
                write=None,
                pool=None
            ),
        )

    async def get_client_with_retries(self) -> AzureOpenAI:
        """Tries to get an OpenAI client using different configurations with retry logic."""
        configurations = self.api_configurations
        random.shuffle(configurations)  # Shuffle the configurations list to randomize

        # Try fetching a client up to max_retries times
        for attempt in range(self.max_retries):
            config = configurations[attempt % len(configurations)]
            try:
                # Try to get a client with the current config
                client = await self.get_openai_client(config['api_key'], config['endpoint'])
                print(f"Successfully obtained client with endpoint: {config['endpoint']}")
                return client
            except Exception as e:
                # Log the error and retry with the next configuration
                print(f"Error while getting client for {config['endpoint']}: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception("Max retries exceeded. Could not obtain a valid OpenAI client.")
        raise Exception("Failed to get OpenAI client after retries.")

    # GPT API interaction logic
    async def gpt_call(self, client, prompt: str, text: str) -> str:
        response = await client.chat.completions.create(
            model=self.engine_name_llm,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]   
        )
        return str(response.choices[0].message.content)

# Agent A: Schedule Generator
class ScheduleGenerator:
    def __init__(self, client: OpenAIClient, constraints, employee_info):
        self.client = client
        self.constraints = constraints
        self.employee_info = employee_info

    async def generate_schedule(self, text) -> t.Dict[str, t.List[str]]:
        client = await self.client.get_client_with_retries()  # Ensure we have a valid client
        # Prompt for generating the schedule
        prompt = (
            "Generate a work schedule for the following employees based on the provided constraints:\n"
            f"Employee Information:\n{self.employee_info}\n"
            f"Constraints:\n{self.constraints}\n"
            "The schedule should be structured as JSON in the following format:\n"
            "{\n"
            "  'Date': ['YYYY-MM-DD', ...],\n"
            "  'Employee1': ['Morning', 'Afternoon', 'Night', 'Off', ...],\n"
            "  'Employee2': ['Morning', 'Afternoon', 'Night', 'Off', ...],\n"
            "}\n"
        )
        schedule = await self.client.gpt_call(client, prompt, text)
        return schedule

# Agent B: Schedule Evaluator
class ScheduleEvaluator:
    def __init__(self, client: OpenAIClient, constraints, employee_info):
        self.client = client
        self.constraints = constraints
        self.employee_info = employee_info

    async def evaluate_schedule(self, schedule: t.Dict[str, t.List[str]]) -> t.Dict[str, t.Any]:
        client = await self.client.get_client_with_retries()  # Ensure we have a valid client
        # Prompt for evaluating the schedule
        prompt = (
            f"Evaluate the following schedule based on the following employees and constraints:\n"
            f"Employee Information:\n{self.employee_info}\n"
            f"Constraints:\n{self.constraints}\n"
            f"Schedule: {schedule}\n"
            "Provide your evaluation in the following JSON format:\n"
            "{\n"
            "  'score': 0-100,  // A score between 0 and 100 for how well the schedule meets the constraints\n"
            "  'feedback': 'string'  // Feedback and suggestions for improvement\n"
            "}"
        )
        evaluation = await self.client.gpt_call(client, prompt, "")
        # Convert the response to a Python dictionary
        return json.loads(evaluation)

# Main system to optimize the schedule
class ScheduleOptimizer:
    def __init__(self, client, constraints, employee_info):
        self.client = client
        self.constraints = constraints
        self.employee_info = employee_info
        self.schedule_generator = ScheduleGenerator(client, constraints, employee_info)
        self.schedule_evaluator = ScheduleEvaluator(client, constraints, employee_info)
        self.schedule_file = "schedule_data.json"

    def load_schedule(self) -> t.Optional[t.Dict[str, t.List[str]]]:
        """Load the existing schedule from a JSON file if it exists."""
        if os.path.exists(self.schedule_file):
            with open(self.schedule_file, 'r') as f:
                return json.load(f)
        return None

    def save_schedule(self, schedule: t.Dict[str, t.List[str]]):
        """Save the schedule to a JSON file."""
        with open(self.schedule_file, 'w') as f:
            json.dump(schedule, f, indent=4)

    async def optimize_schedule(self, initial_text: str, expected_score: int = 90, max_iterations=5) -> t.Dict[str, t.List[str]]:
        # Load existing schedule if it exists
        existing_schedule = self.load_schedule()
        if existing_schedule:
            print("Existing schedule found. Using as seed for optimization.")
            initial_text = json.dumps(existing_schedule)
        else:
            print("No existing schedule found. Generating a new one.")

        feedback_list = []  # To keep track of the last two feedbacks
        best_score = -1  # Track the best score so far
        best_schedule = None  # Track the best schedule so far

        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}:")

            # Step 1: Agent A generates the initial schedule
            schedule = await self.schedule_generator.generate_schedule(initial_text)
            print(f"Generated Schedule:\n{schedule}\n")

            # Step 2: Agent B evaluates the schedule
            evaluation = await self.schedule_evaluator.evaluate_schedule(schedule)
            print(f"Evaluation and Suggestions:\n{evaluation['feedback']}\n")

            score = evaluation.get('score', 0)
            print(f"Score: {score}")

            # Update best schedule only if the current score is higher
            if score > best_score:
                best_score = score
                best_schedule = schedule
                print("New best schedule found with higher score.")

            if score >= expected_score:
                print("Optimized schedule found.")
                self.save_schedule(best_schedule)  # Save the final best schedule
                break
            else:
                print("Revising the schedule based on feedback...\n")
                # Keep only the last two feedbacks
                feedback_list.append(evaluation['feedback'])
                if len(feedback_list) > 2:
                    feedback_list.pop(0)  # Remove the oldest feedback
                
                # Prepare the new input text using the last two feedbacks
                initial_text = "\n".join(feedback_list)

        # Save the best schedule found during the iterations
        if best_schedule:
            self.save_schedule(best_schedule)
            self.visualize_schedule(best_schedule)  # Visualize the best schedule at the end
        return best_schedule

    def visualize_schedule(self, schedule: str):
        """Convert the final schedule (in JSON string format) to a DataFrame and plot it."""
        try:
            # Convert the JSON string schedule into a dictionary
            schedule_dict = json.loads(schedule)

            # Create a DataFrame from the dictionary
            df = pd.DataFrame(schedule_dict)

            # Call the plot function to visualize the schedule
            plot_schedule(df)
        except ValueError as e:
            print(f"Error parsing the schedule JSON: {e}")

# Schedule plotting function
def plot_schedule(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Generate the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Set font size and scaling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Set color coding (e.g., yellow for Off days)
    for i in range(len(df)):
        for j in range(1, len(df.columns)):
            if df.iloc[i, j] == 'Off':
                table[(i+1, j)].set_facecolor('#FFFF99')  # Yellow for Off days
            elif df.iloc[i, j] == 'Night':
                table[(i+1, j)].set_facecolor('#556B2F')  # Dark Green for Night shifts
            elif df.iloc[i, j] == 'Afternoon':
                table[(i+1, j)].set_facecolor('#C0C0C0')  # Gray for Afternoon shifts
            elif df.iloc[i, j] == 'Morning':
                table[(i+1, j)].set_facecolor('#87CEEB')  # Light Blue for Morning shifts

    plt.title('Employee Schedule', fontsize=16)
    plt.show()

# Example constraints for the schedule
constraints = constraints = """
1. Night Shift Limitation:
   - Each person can be assigned no more than 2 night shifts per week.
   - Penalty: For each additional night shift beyond the limit, a penalty score of 100 points will be applied.

2. Balanced Shift Distribution:
   - Shifts (morning, afternoon, night) should be distributed as evenly as possible for each person, with no significant bias toward a particular shift type.
   - Penalty: If any shift type (morning, afternoon, night) is assigned more than 1 time above or below the others, a penalty score of 30 points will be applied.

3. Rest Days Limitation:
    - Each person should only have 1 rest day per week.
    - Penalty: For each additional rest day beyond the limit, a penalty score of 100 points will be applied.


4. FINAL GOAL: Ensure each day is covered:
    - At least one person should be available for each shift (morning, afternoon, night) on each day.
    - Penalty: For each missing shift coverage, a penalty score of 150 points will be applied.
"""



# Employee information (e.g., names and roles)
employee_info = """
Employees:
1. Zhang KeKe: 
   - Role: Senior Nurse
   - Prefers morning shifts in two days
   - Available from Monday to Friday
   - Has experience in handling critical patients

2. Li MeiMei: 
   - Role: Junior Doctor
   - Prefers night shifts in two days
   - Available from Monday to Saturday
   - Has expertise in pediatric care
   - Cannot work two consecutive night shifts

3. Han TiaoDuo: 
   - Role: Senior Doctor
   - Prefers afternoon shifts in two days
   - Available from Monday to Friday
   - Specializes in surgery, often required in emergencies
   - Needs at least 2 consecutive rest days after 5 working days

4. Li DongDong: 
   - Role: General Practitioner
   - No specific shift preference
   - Available from Wednesday to Sunday
   - Flexible with shifts but must avoid back-to-back shifts without a break

5. Zhang HaoHao: 
   - Role: Junior Nurse
   - No specific shift preference
   - Available from Tuesday to Saturday
   - Recently trained in operating room procedures, but needs supervision

6. Li Lei: 
   - Role: Senior Nurse
   - No specific shift preference
   - Available all week
   - Highly experienced in ICU care, often on call for critical situations
   - Can handle back-to-back night shifts, but needs a day off after 3 consecutive shifts
"""

# Client instantiation (replace with real OpenAI client setup)
client = OpenAIClient()

# Optimizer system
optimizer = ScheduleOptimizer(client, constraints, employee_info)

# Run the optimization (replace with actual asyncio event loop if necessary)
optimized_schedule = asyncio.run(optimizer.optimize_schedule("Initial schedule request"))
print(f"Final Optimized Schedule:\n{optimized_schedule}")
