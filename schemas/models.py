from pydantic import BaseModel, Field
from typing import List

 #--- Knowledge Graph Schema ---
class Triple(BaseModel):
  """Represents a single knowledge graph triple (Subject-Predicate-Object)."""
  subject: str = Field(description="The entity or concept being described.")
  predicate: str = Field(description="The relationship or property connecting the subject and object.")
  object: str = Field(description="The entity, concept, or value related to the subject via the predicate.")

class TripleList(BaseModel):
    """A list of knowledge graph triples."""
    triples: List[Triple]

