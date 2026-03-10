from jobs import JobCategory, JOB_CATEGORY_DESCRIPTIONS

_categories_with_desc = "\n".join(
    f"- {c.value}: {JOB_CATEGORY_DESCRIPTIONS[c]}" for c in JobCategory
)

JOB_DESCRIPTION_PROMPT = (
    "Based on user description determine which job category this user should be assigned. One person can have multiple job categories. "
    "Return array of objects containing index from file of given person and array of tags.\n\n"
    f"Available job categories:\n{_categories_with_desc}"
)

JOB_DESCRIPTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "job_description_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "IT",
                                        "transport",
                                        "edukacja",
                                        "medycyna",
                                        "praca z ludźmi",
                                        "praca z pojazdami",
                                        "praca fizyczna",
                                    ],
                                },
                            },
                        },
                        "required": ["index", "tags"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["people"],
            "additionalProperties": False,
        },
    },
}
