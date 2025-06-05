import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.9.23"
    application
    kotlin("plugin.serialization") version "1.9.23"
}

group = "com.example.rag"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    // Kotlinx Serialization for JSON
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

    // Ktor Client for HTTP requests
    implementation("io.ktor:ktor-client-core:2.3.12")
    implementation("io.ktor:ktor-client-cio:2.3.12") // CIO engine for Ktor
    implementation("io.ktor:ktor-client-content-negotiation:2.3.12")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.12")

    // Jsoup for HTML scraping
    implementation("org.jsoup:jsoup:1.17.2")

    // Dotenv for loading .env files
    implementation("io.github.cdimascio:dotenv-kotlin:6.4.1")
    
    // Logback for better logging (optional but good practice)
    implementation("ch.qos.logback:logback-classic:1.4.14")

    // Changed jtokkit version
    implementation("com.knuddels:jtokkit:0.6.1")

}

// Configure the application plugin
application {
    mainClass.set("com.example.rag.ChatCliKt")
}

// Custom task to run the IndexWebsite script
tasks.register<JavaExec>("runIndex") {
    group = "application"
    description = "Runs the IndexWebsite.kt script to scrape and index a website."
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("com.example.rag.IndexWebsiteKt")
    standardOutput = System.out 
}

// Custom task to run the ChatCli script (can also use the default 'run' task)
tasks.register<JavaExec>("runChat") {
    group = "application"
    description = "Runs the ChatCli.kt script for interactive chat."
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("com.example.rag.ChatCliKt")
    standardInput = System.`in`
    standardOutput = System.out
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}
