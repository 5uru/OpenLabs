from pydantic import BaseModel, Field
from typing import List, Optional



class AfricanPharmacopoeiaRecipe(BaseModel):
    """Structured model for African traditional medicine recipes with optional fields"""

    recipe_name: str = Field(
            description="Common name of the recipe (e.g., 'Fever remedy')."
    )

    plant_common_name: Optional[List[str]] = Field(
            default=None,
            description="List of local plant names (e.g., ['neem', 'bark'])."
    )

    scientific_name: List[str] = Field(
            default=None,
            description="List of scientific plant names (e.g., ['Azadirachta indica'])."
    )

    diseases_treated: List[str] = Field(
            description="List of diseases/symptoms treated (e.g., ['malaria', 'fever'])."
    )

    preparation_steps: List[str] = Field(
            default=None,
            description="Simple preparation steps with quantities/times (e.g., ['Crush 5 leaves', 'Mix 1 cup water', 'Boil 10 min'])."
    )

    dosage_instructions: List[str] = Field(
            default=None,
            description="Simple dosage instructions (e.g., ['1 cup morning', '1 tsp every 4 hours'])."
    )

    preparation_time: List[str] = Field(
            default=None,
            description="Total preparation time (e.g., '15 min', '2 hours')."
    )


    additional_notes: List[str] = Field(
            default=None,
            description="Any other important notes (e.g., 'Do not use during pregnancy')."
    )



class AfricanPharmacopoeiaRecipeList(BaseModel):
    pharmacopoeia_recipe: list[AfricanPharmacopoeiaRecipe]