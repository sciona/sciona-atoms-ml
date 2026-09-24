"""Source population order, global feature selection and model-slot naming."""
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.model_selection.named_regression import witness_materialized

POPULATIONS = ['katl','kclt','kden','kdfw','kjfk','kmem','kmia','kord','kphx','ksea']


@register_atom(witness=witness_materialized)
def training_tables(population_tables: dict, target_name: str) -> tuple:
    """Expose ten source-ordered local tables and explicit global table selections."""
    if set(population_tables) != set(POPULATIONS) or target_name != 'minutes_until_pushback':
        raise ValueError('Complete source population tables and target required')
    tables = [population_tables[label] for label in POPULATIONS]
    columns = [['gufi','timestamp',target_name]+[c for c in table.columns if 'mfs' in c or 'etd' in c] for table in tables]
    return (*tables,tables,list(POPULATIONS),columns,'feat_cat_airport','timestamp')


@register_atom(witness=witness_materialized)
def model_bindings(residual0: object, direct0: object, residual1: object, direct1: object,
        residual2: object, direct2: object, residual3: object, direct3: object,
        residual4: object, direct4: object, residual5: object, direct5: object,
        residual6: object, direct6: object, residual7: object, direct7: object,
        residual8: object, direct8: object, residual9: object, direct9: object, global_model: object) -> tuple:
    """Name 21 distinct fits and bind 30 local slots plus one shared global state."""
    values=[residual0,direct0,residual1,direct1,residual2,direct2,residual3,direct3,residual4,direct4,
        residual5,direct5,residual6,direct6,residual7,direct7,residual8,direct8,residual9,direct9,global_model]
    if any(value is None for value in values) or len({id(value) for value in values}) != 21:
        raise ValueError('Twenty-one independent source fit objects required')
    models={str(i):value for i,value in enumerate(values)}
    bindings={label.upper():{0:str(2*i),1:str(2*i+1),2:str(2*i)} for i,label in enumerate(POPULATIONS)}
    return models,bindings,{'global_model':'20'}
