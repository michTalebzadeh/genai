BEGIN {
    FS = "./"  # Field separator
}

{
    # Split size into value and unit
    split($1, a, /[MG]/)
    val = a[1]
    unit = a[2]

    # Convert value to megabytes
    if (unit == "G") {
        val = val * 1024
    }

    # Extract major version
    major = substr($2, 1, 1)

    # Accumulate sizes
    sizes[major] += val
    total += val
}

END {
    PROCINFO["sorted_in"] = "@ind_num_asc"  # Sort major versions numerically

    # Print results
    for (major in sizes) {
        printf("Major Release %s: %.2f MB\n", major, sizes[major])
    }
    printf("Total: %.2f MB\n", total)
}

