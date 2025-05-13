# Environment Info

## Unit Type Table

Default uses `VERSION_ORIGINAL`

Note: bold is the difference to the previous chart

### VERSION_ORIGINAL

| Attribute | Resource | Base | Barracks | Worker | Light | Heavy | Ranged |
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| cost | x | 10 | 5 | 1 | 2 | 2 | 2 |
| hp | x | 10 | 4 | 1 | 4 | 4 | 1 |
| minDamage | x | x | x | 1 | 2 | 4 | 1 |
| maxDamage | x | x | x | 1 | 2 | 4 | 1 |
| attackRange | x | x | x | 1 | 1 | 1 | 3 |
| produceTime | x | 250 | 200 | 50 | 80 | 120 | 100 |
| moveTime | x | x | x | 10 | 8 | 12 | 10 |
| attackTime | x | x | x | 5 | 5 | 5 | 5 |
| harvestTime | x | x | x | 20 | x | x | x |
| returnTime | x | x | x | 10 | x | x | x |
| isResource | true | false | false | false | false | false | false |
| isStockpile | false | true | false | false | false | false | false |
| canHarvest | false | false | false | true | false | false | false |
| canMove | false | false| false | true | true | true | true |
| canAttack | false | false| false | true | true | true | true |
| sightRadius | 0 | 5 | 3 | 3 | 2 | 2 | 3 |

### VERSION_ORIGINAL_FINETUNED

| Attribute | Resource | Base | Barracks | Worker | Light | Heavy | Ranged |
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| cost | x | 10 | 5 | 1 | 2 | **3** | 2 |
| hp | x | 10 | 4 | 1 | 4 | **8** | 1 |
| minDamage | x | x | x | 1 | 2 | 4 | 1 |
| maxDamage | x | x | x | 1 | 2 | 4 | 1 |
| attackRange | x | x | x | 1 | 1 | 4 | 3 |
| produceTime | x | **200** | **100** | 50 | 80 | 120 | 100 |
| moveTime | x | x | x | 10 | 8 | **10** | 10 |
| attackTime | x | x | x | 5 | 5 | 5 | 5 |
| harvestTime | x | x | x | 20 | x | x | x |
| returnTime | x | x | x | 10 | x | x | x |
| isResource | true | false | false | false | false | false | false |
| isStockpile | false | true | false | false | false | false | false |
| canHarvest | false | false | false | true | false | false | false |
| canMove | false | false| false | true | true | true | true |
| canAttack | false | false| false | true | true | true | true |
| sightRadius | 0 | 5 | 3 | 3 | 2 | 2 | 3 |

### VERSION_NON_DETERMINISTIC

| Attribute | Resource | Base | Barracks | Worker | Light | Heavy | Ranged |
|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| cost | x | 10 | 5 | 1 | 2 | 3 | 2 |
| hp | x | 10 | 4 | 1 | 4 | 8 | 1 |
| minDamage | x | x | x | **0** | **1** | **0** | 1 |
| maxDamage | x | x | x | **2** | **3** | **6** | **2** |
| attackRange | x | x | x | 1 | 1 | 4 | 3 |
| produceTime | x | 200 | 100 | 50 | 80 | 120 | 100 |
| moveTime | x | x | x | 10 | 8 | 10 | 10 |
| attackTime | x | x | x | 5 | 5 | 5 | 5 |
| harvestTime | x | x | x | 20 | x | x | x |
| returnTime | x | x | x | 10 | x | x | x |
| isResource | true | false | false | false | false | false | false |
| isStockpile | false | true | false | false | false | false | false |
| canHarvest | false | false | false | true | false | false | false |
| canMove | false | false| false | true | true | true | true |
| canAttack | false | false| false | true | true | true | true |
| sightRadius | 0 | 5 | 3 | 3 | 2 | 2 | 3 |