from enum import Enum


class JobCategory(Enum):
    IT = "IT"
    TRANSPORT = "transport"
    EDUKACJA = "edukacja"
    MEDYCYNA = "medycyna"
    PRACA_Z_LUDZMI = "praca z ludźmi"
    PRACA_Z_POJAZDAMI = "praca z pojazdami"
    PRACA_FIZYCZNA = "praca fizyczna"


JOB_CATEGORY_DESCRIPTIONS = {
    JobCategory.IT: "programowanie, bazy danych, algorytmy, systemy informatyczne",
    JobCategory.TRANSPORT: "logistyka, przewóz towarów, planowanie tras, spedycja",
    JobCategory.EDUKACJA: "nauczanie, przekazywanie wiedzy, praca z uczniami",
    JobCategory.MEDYCYNA: "leczenie, diagnostyka, opieka zdrowotna, farmacja",
    JobCategory.PRACA_Z_LUDZMI: "prawo, bezpieczeństwo, mediacja, pomoc społeczna",
    JobCategory.PRACA_Z_POJAZDAMI: "mechanika samochodowa, naprawa pojazdów, serwis",
    JobCategory.PRACA_FIZYCZNA: "rzemiosło, budownictwo, instalacje, prace manualne",
}
