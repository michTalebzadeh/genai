{
    split($1, a, /[MG]/)
    val = a[1]
    unit = a[2]
    if (unit == "G") {
        val = val * 1024
    }
    sum += val
}
END {
    printf("%.2f GB\n", sum / 1024)
}

