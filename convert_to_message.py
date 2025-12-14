import json

INPUT_PATH = "final_lora_dataset.jsonl"      # your curated dataset
OUTPUT_PATH = "lora_messages_dataset.jsonl"  # new file for LoRA training


def build_assistant_text(output_obj):
    """
    Turn {role, experience, keywords[]} into a single assistant string.
    Example:
      role: Python Developer
      experience: 1 years
      keywords: Python, Sqlite, django basics, rest integration
    """
    role = output_obj.get("role", "")
    experience = output_obj.get("experience", "")
    keywords = output_obj.get("keywords", [])

    # join keywords as comma-separated string
    kw_str = ", ".join(keywords)

    text = (
        f"role: {role}\n"
        f"experience: {experience}\n"
        f"keywords: {kw_str}"
    )
    return text


def convert_dataset(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    count_in = 0
    count_out = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            count_in += 1
            obj = json.loads(line)

            user_input = obj.get("input", "")
            output_obj = obj.get("output", {})

            assistant_text = build_assistant_text(output_obj)

            chat_example = {
                "messages": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_text}
                ]
            }

            outfile.write(json.dumps(chat_example) + "\n")
            count_out += 1

    print(f"Done ✅ Converted {count_in} examples → {count_out} chat-style examples.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    convert_dataset()

