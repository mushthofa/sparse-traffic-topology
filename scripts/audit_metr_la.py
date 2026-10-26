"""Audit local METR-LA data and graph without constructing benchmark targets."""

from prepare_metr_la import main

if __name__ == "__main__":
    main(prepare=False)
